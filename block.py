"""
Core SQL Agent Classes for Excel/CSV Data Querying
Contains FilesDatabaseManager and FilesSQLAgent classes

Pipeline (v4 — with clarification + suspicious-result detection):
  check_clarification_needed          ← bail early if question is ambiguous
  → parse_intent
  → select_relevant_schema
  → build_query_plan
  → generate_sql_from_plan
  → validate_sql (deterministic)
  → execute + repair loop (up to 2 repairs)
  → analyze_result            ← NEW: detect zero/null/empty/entity-mismatch
  → generate_answer_summary   ← calibrated to ResultFlags
  → QueryResponse / ClarificationRequest
"""

# ============================================================================
# 1. Setup Environment
# ============================================================================

import os
import re
import json
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import (
    ClientSecretCredential,
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Load .env from the same directory as THIS file — reliable regardless of cwd.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH, override=True)

# Auth debug logging (True/False only — no secrets printed)
print(f"[sql_agent_core] .env path         : {_ENV_PATH}")
print(f"[sql_agent_core] .env exists        : {_ENV_PATH.exists()}")
print(f"[sql_agent_core] AZURE_CLIENT_ID    : {bool(os.getenv('AZURE_CLIENT_ID'))}")
print(f"[sql_agent_core] AZURE_TENANT_ID    : {bool(os.getenv('AZURE_TENANT_ID'))}")
print(f"[sql_agent_core] AZURE_CLIENT_SECRET: {bool(os.getenv('AZURE_CLIENT_SECRET'))}")


# ============================================================================
# 2. Dataclasses
# ============================================================================

@dataclass
class AzureConfig:
    endpoint: str
    deployment_name: str
    api_version: str


@dataclass
class ClarificationRequest:
    """Returned when the question is too ambiguous to safely generate SQL."""
    ambiguity_type: str          # ambiguous_metric | fuzzy_entity | missing_filter_value | non_trivial_assumption | mixed
    ambiguous_term: str          # exact phrase in the question that triggered this
    clarification_question: str
    clarification_reason: str
    clarification_options: List[str]
    option_labels: List[str] = field(default_factory=list)  # parallel to options; short interpretation labels
    secondary_note: str = ""     # follow-up note about a residual ambiguity after the main one is resolved


@dataclass
class ParsedIntent:
    """Structured representation of what the user is asking."""
    action: str                 # aggregate | filter | list | count | compare | lookup
    entities: List[str]
    filters: Dict[str, Any]
    aggregation: str            # sum | count | avg | max | min | none
    group_by_hint: Optional[str]
    sort_hint: Optional[str]
    sort_order: str
    limit: Optional[int]
    raw: Dict = field(default_factory=dict)


@dataclass
class QueryPlan:
    """Intermediate plan produced before SQL generation."""
    tables: List[str]
    columns: List[str]
    filters: List[Dict]
    aggregation: Optional[Dict]
    group_by: List[str]
    order_by: Optional[Dict]
    limit: Optional[int]
    joins: List[Dict]
    notes: str = ""


@dataclass
class QueryResponse:
    """Richer response object returned by run_query()."""
    question: str
    interpreted_intent: Dict
    relevant_tables: List[str]
    query_plan: Dict
    sql_query: Optional[str]
    results: Optional[pd.DataFrame]
    answer_summary: str
    warnings: List[str]
    trace: List[str]
    success: bool
    error: Optional[str] = None
    repair_attempts: int = 0


@dataclass
class ResultFlags:
    """
    Structured flags from post-execution result analysis.
    All fields are False / empty by default (non-suspicious result).
    """
    empty_result: bool = False               # result set has zero rows
    suspicious_zero_result: bool = False     # aggregate returned 0 or null
    possible_exact_match_miss: bool = False  # exact-match filter on text col + zero/empty
    entity_match_uncertain: bool = False     # similar values found in DB that were not matched
    similar_values: List[str] = field(default_factory=list)  # LIKE-found alternatives
    filter_column: str = ""                  # the column that triggered the concern
    filter_value: str = ""                   # the value that was searched


# ============================================================================
# 3. SQL safety constants
# ============================================================================

_DANGEROUS_KEYWORDS = {
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE",
    "TRUNCATE", "EXEC", "EXECUTE", "PRAGMA", "ATTACH", "DETACH",
    "REPLACE", "MERGE", "CALL", "GRANT", "REVOKE",
}


# ============================================================================
# 4. Azure OpenAI Client
# ============================================================================

def _build_credential():
    """
    Primary  — ClientSecretCredential when all three service-principal vars exist.
    Fallback — DefaultAzureCredential for managed-identity / interactive scenarios.
    """
    client_id     = os.getenv("AZURE_CLIENT_ID")
    tenant_id     = os.getenv("AZURE_TENANT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")

    if client_id and tenant_id and client_secret:
        print("[sql_agent_core] Auth path: ClientSecretCredential (explicit env vars)")
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

    print("[sql_agent_core] Auth path: DefaultAzureCredential (fallback)")
    return DefaultAzureCredential()


def initialize_azure_client(endpoint: str, deployment: str, api_version: str):
    """Initialize Azure OpenAI client with deterministic credential selection."""
    try:
        credential     = _build_credential()
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
        )
        config = AzureConfig(
            endpoint=endpoint,
            deployment_name=deployment,
            api_version=api_version,
        )
        return client, config
    except Exception as e:
        raise Exception(f"Error initializing Azure OpenAI client: {e}")


# ============================================================================
# 5. DATABASE MANAGER  (unchanged)
# ============================================================================

class FilesDatabaseManager:
    def __init__(self):
        self.engine       = None
        self.connection   = None
        self.tables_info: Dict = {}
        self.loaded_files: List = []

    def load_file(self, file_path: str, sheet_names: List[str] = None,
                  table_name: str = None) -> bool:
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.xlsx', '.xls']:
                return self.load_excel_file(file_path, sheet_names)
            elif ext == '.csv':
                return self.load_csv_file(file_path, table_name)
            else:
                print(f"Unsupported file format: {ext}")
                return False
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return False

    def _detect_csv_delimiter(self, file_path: str, encoding: str) -> str:
        delimiters = [',', ';', '\t', '|']
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
            counts = {d: first_line.count(d) for d in delimiters}
            valid  = {d: c for d, c in counts.items() if c > 0}
            if valid:
                return max(valid, key=valid.get)
        except Exception:
            pass
        return ','

    def load_csv_file(self, file_path: str, table_name: str = None) -> bool:
        try:
            if not os.path.exists(file_path):
                print(f"CSV file not found: {file_path}")
                return False
            if self.engine is None:
                self.engine = create_engine(
                    'sqlite:///:memory:',
                    connect_args={'check_same_thread': False}
                )
                self.connection = self.engine.connect()

            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            for enc in encodings:
                try:
                    delim = self._detect_csv_delimiter(file_path, enc)
                    df    = pd.read_csv(file_path, encoding=enc, delimiter=delim)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None or df.empty:
                print(f"Could not read CSV or file is empty: {os.path.basename(file_path)}")
                return False

            df.columns = [self._clean_column_name(c) for c in df.columns]
            if table_name is None:
                table_name = self._clean_table_name(
                    os.path.splitext(os.path.basename(file_path))[0]
                )
            else:
                table_name = self._clean_table_name(table_name)

            df.to_sql(table_name, self.connection, if_exists='replace', index=False)
            self.tables_info[table_name] = {
                'source_file':  os.path.basename(file_path),
                'source_type':  'CSV',
                'columns':      list(df.columns),
                'row_count':    len(df),
                'column_types': df.dtypes.to_dict(),
            }
            self.loaded_files.append({
                'file_path': file_path,
                'type':      'CSV',
                'tables':    [f"{os.path.basename(file_path)} -> {table_name}"],
            })
            return True
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False

    def load_excel_file(self, file_path: str, sheet_names: List[str] = None) -> bool:
        try:
            if not os.path.exists(file_path):
                print(f"Excel file not found: {file_path}")
                return False
            if sheet_names is None:
                ef         = pd.ExcelFile(file_path)
                sheet_names = ef.sheet_names
                ef.close()
            if self.engine is None:
                self.engine = create_engine(
                    'sqlite:///:memory:',
                    connect_args={'check_same_thread': False}
                )
                self.connection = self.engine.connect()

            loaded = []
            for sheet in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    if df.empty:
                        continue
                    df.columns = [self._clean_column_name(c) for c in df.columns]
                    tname      = self._clean_table_name(sheet)
                    df.to_sql(tname, self.connection, if_exists='replace', index=False)
                    self.tables_info[tname] = {
                        'source_file':  os.path.basename(file_path),
                        'source_sheet': sheet,
                        'columns':      list(df.columns),
                        'row_count':    len(df),
                        'column_types': df.dtypes.to_dict(),
                    }
                    loaded.append(f"{sheet} -> {tname}")
                except Exception as e:
                    print(f"Failed to load sheet '{sheet}': {e}")

            if loaded:
                self.loaded_files.append({
                    'file_path': file_path,
                    'type':      'Excel',
                    'sheets':    loaded,
                })
                return True
            return False
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return False

    def _clean_column_name(self, col: str) -> str:
        col = str(col)
        if col in ('nan', 'NaN'):
            col = 'unnamed_column'
        col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        if col[0].isdigit():
            col = 'col_' + col
        return col.lower()

    def _clean_table_name(self, name: str) -> str:
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if name[0].isdigit():
            name = 'table_' + name
        return name.lower()

    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            return pd.read_sql_query(query, self.connection)
        except Exception as e:
            raise Exception(f"Query execution failed: {e}")

    def get_schema_info(self) -> Dict:
        schema = {}
        for tname, info in self.tables_info.items():
            cols = []
            for col in info['columns']:
                dtype = str(info['column_types'].get(col, 'TEXT'))
                cols.append({'name': col, 'type': dtype, 'nullable': True})
            schema[tname] = cols
        return schema

    def get_tables_summary(self) -> pd.DataFrame:
        if not self.tables_info:
            return pd.DataFrame()
        rows = []
        for tname, info in self.tables_info.items():
            stype = info.get('source_type', 'Excel')
            source = (
                f"{info['source_file']} (CSV)"
                if stype == 'CSV'
                else f"{info['source_file']} - Sheet: {info.get('source_sheet', '?')}"
            )
            rows.append({
                'Table':   tname,
                'Source':  source,
                'Rows':    info['row_count'],
                'Columns': len(info['columns']),
            })
        return pd.DataFrame(rows)

    def disconnect(self):
        if self.connection:
            self.connection.close()


# ============================================================================
# 6. SQL AGENT  (v3 — clarification + full pipeline)
# ============================================================================

class FilesSQLAgent:
    def __init__(self, azure_client, files_db: FilesDatabaseManager,
                 deployment_name: str):
        self.azure_client    = azure_client
        self.files_db        = files_db
        self.deployment_name = deployment_name
        self.schema_info     = self.files_db.get_schema_info()
        self.conversation_history: List[Dict] = []
        self.last_query_result   = None
        self.last_query_context  = None

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def add_to_history(self, question: str, result: Dict):
        self.conversation_history.append({
            'question':       question,
            'sql_query':      result.get('sql_query'),
            'result_summary': (
                f"Returned {len(result.get('results', []))} rows"
                if result.get('success') else "Query failed"
            ),
        })
        self.last_query_result  = result.get('results')
        self.last_query_context = {
            'previous_question': question,
            'previous_sql':      result.get('sql_query'),
            'row_count':         len(result.get('results', [])) if result.get('success') else 0,
        }

    def get_conversation_context(self) -> str:
        if not self.conversation_history:
            return ""
        ctx = "\nCONVERSATION HISTORY (for follow-up questions):\n"
        for i, entry in enumerate(self.conversation_history[-3:], 1):
            ctx += f"\n{i}. Q: {entry['question']}\n"
            ctx += f"   SQL: {entry['sql_query']}\n"
            ctx += f"   Result: {entry['result_summary']}\n"
        return ctx

    @staticmethod
    def _quote_ident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def _escape_sql_literal(value: Any) -> str:
        return str(value).replace("'", "''")

    def _table_exists(self, table_name: str) -> bool:
        return table_name in self.files_db.tables_info

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        info = self.files_db.tables_info.get(table_name, {})
        return column_name in info.get('columns', [])

    def _build_guided_overview(self, question: str) -> Dict:
        tables_info = self.files_db.tables_info or {}
        total_tables = len(tables_info)
        total_rows = int(sum(int(info.get('row_count', 0)) for info in tables_info.values()))
        total_columns = int(sum(len(info.get('columns', [])) for info in tables_info.values()))

        results = pd.DataFrame([{
            'total_tables': total_tables,
            'total_rows': total_rows,
            'total_columns': total_columns,
        }])
        answer_summary = (
            f"Loaded tables: {total_tables}. "
            f"Total rows across loaded tables: {total_rows:,}. "
            f"Total columns (summed across tables): {total_columns:,}."
        )
        pseudo_sql = "-- guided deterministic overview from tables_info"

        self.add_to_history(question, {
            'sql_query': pseudo_sql,
            'results': results,
            'success': True,
        })
        return {
            'question': question,
            'sql_query': pseudo_sql,
            'results': results,
            'success': True,
            'error': None,
            'answer_summary': answer_summary,
            'warnings': [],
            'trace': [
                'Guided route: dataset overview',
                'Computed totals from files_db.tables_info',
            ],
            'interpreted_intent': {'route': 'guided_overview'},
            'relevant_tables': list(tables_info.keys()),
            'query_plan': {'route': 'guided_overview'},
            'repair_attempts': 0,
            'needs_clarification': False,
        }

    def _build_guided_tab_breakdown(self, question: str, sample_values_per_column: int = 3) -> Dict:
        """Return per-table column summaries with dtype and sample values."""
        tables_info = self.files_db.tables_info or {}
        rows: List[Dict[str, Any]] = []

        sample_limit = max(1, min(int(sample_values_per_column), 5))
        for table_name, info in tables_info.items():
            columns = list(info.get('columns', []))
            col_types = info.get('column_types', {})

            for col in columns:
                dtype = str(col_types.get(col, 'TEXT'))
                sample_text = ""
                try:
                    q_table = self._quote_ident(table_name)
                    q_col = self._quote_ident(col)
                    sample_df = self.files_db.execute_query(
                        f"SELECT DISTINCT TRIM(CAST({q_col} AS TEXT)) AS v "
                        f"FROM {q_table} "
                        f"WHERE {q_col} IS NOT NULL AND TRIM(CAST({q_col} AS TEXT)) <> '' "
                        f"LIMIT {sample_limit}"
                    )
                    values = [
                        str(v)
                        for v in sample_df.get('v', pd.Series(dtype='object')).tolist()
                        if str(v) not in ('nan', 'None', '')
                    ]
                    sample_text = ", ".join(values[:sample_limit])
                except Exception:
                    sample_text = ""

                rows.append({
                    'table_name': table_name,
                    'row_count': int(info.get('row_count', 0)),
                    'column_name': str(col),
                    'column_type': dtype,
                    'sample_values': sample_text,
                })

        results = pd.DataFrame(rows)
        answer_summary = (
            f"Prepared column breakdown for {len(tables_info)} loaded table(s)"
            + (f" across {len(results)} column entries." if not results.empty else ".")
        )
        pseudo_sql = "-- guided deterministic tab breakdown from tables_info + sample lookups"

        self.add_to_history(question, {
            'sql_query': pseudo_sql,
            'results': results,
            'success': True,
        })
        return {
            'question': question,
            'sql_query': pseudo_sql,
            'results': results,
            'success': True,
            'error': None,
            'answer_summary': answer_summary,
            'warnings': [],
            'trace': [
                'Guided route: tab breakdown',
                'Computed per-table column summaries with sample values',
            ],
            'interpreted_intent': {'route': 'guided_tab_breakdown'},
            'relevant_tables': list(tables_info.keys()),
            'query_plan': {
                'route': 'guided_tab_breakdown',
                'sample_values_per_column': sample_limit,
            },
            'repair_attempts': 0,
            'needs_clarification': False,
        }

    def execute_guided_query(self, guided_request: Dict[str, Any]) -> Dict:
        """Execute deterministic guided queries without LLM SQL generation."""
        intent = str(guided_request.get('intent', '')).strip().lower()
        question = str(guided_request.get('question') or 'Guided query').strip()

        if intent == 'overview':
            return self._build_guided_overview(question)
        if intent == 'tab_breakdown':
            return self._build_guided_tab_breakdown(
                question,
                sample_values_per_column=int(guided_request.get('sample_values_per_column') or 3),
            )

        table_name = str(guided_request.get('table_name') or '').strip()
        if not table_name or not self._table_exists(table_name):
            return {
                'question': question,
                'sql_query': None,
                'results': None,
                'success': False,
                'error': 'Invalid or missing table for guided query.',
                'answer_summary': '',
                'warnings': [],
                'trace': ['Guided route failed: invalid table'],
                'interpreted_intent': {'route': f'guided_{intent}'},
                'relevant_tables': [],
                'query_plan': {},
                'repair_attempts': 0,
                'needs_clarification': False,
            }

        agg = str(guided_request.get('aggregation', 'sum')).strip().lower()
        if agg not in {'sum', 'avg', 'count', 'min', 'max'}:
            agg = 'sum'

        value_column = str(guided_request.get('value_column') or '__rows__').strip()
        group_by_column = str(guided_request.get('group_by_column') or '').strip()
        where_column = str(guided_request.get('where_column') or '').strip()
        where_value = str(guided_request.get('where_value') or '').strip()
        where_mode = str(guided_request.get('where_mode') or 'contains').strip().lower()
        limit = int(guided_request.get('limit') or 20)
        limit = max(1, min(limit, 500))

        # Build aggregate expression
        if agg == 'count' and value_column in {'', '__rows__', '*'}:
            agg_expr = 'COUNT(*)'
        else:
            if not value_column or not self._column_exists(table_name, value_column):
                return {
                    'question': question,
                    'sql_query': None,
                    'results': None,
                    'success': False,
                    'error': 'Invalid or missing value column for guided query.',
                    'answer_summary': '',
                    'warnings': [],
                    'trace': ['Guided route failed: invalid value column'],
                    'interpreted_intent': {'route': f'guided_{intent}'},
                    'relevant_tables': [table_name],
                    'query_plan': {},
                    'repair_attempts': 0,
                    'needs_clarification': False,
                }
            agg_expr = f"{agg.upper()}({self._quote_ident(value_column)})"

        # Optional where clause
        where_clause = ''
        if where_column and where_value:
            if not self._column_exists(table_name, where_column):
                return {
                    'question': question,
                    'sql_query': None,
                    'results': None,
                    'success': False,
                    'error': 'Invalid filter column for guided query.',
                    'answer_summary': '',
                    'warnings': [],
                    'trace': ['Guided route failed: invalid filter column'],
                    'interpreted_intent': {'route': f'guided_{intent}'},
                    'relevant_tables': [table_name],
                    'query_plan': {},
                    'repair_attempts': 0,
                    'needs_clarification': False,
                }
            safe_value = self._escape_sql_literal(where_value)
            q_filter_col = self._quote_ident(where_column)
            if where_mode == 'equals':
                where_clause = f" WHERE TRIM(CAST({q_filter_col} AS TEXT)) = TRIM('{safe_value}')"
            else:
                where_clause = f" WHERE CAST({q_filter_col} AS TEXT) LIKE '%{safe_value}%'"

        q_table = self._quote_ident(table_name)
        metric_alias = self._quote_ident('metric_value')

        if intent == 'aggregate':
            sql_query = f"SELECT {agg_expr} AS {metric_alias} FROM {q_table}{where_clause}"
        elif intent == 'group_by':
            if not group_by_column or not self._column_exists(table_name, group_by_column):
                return {
                    'question': question,
                    'sql_query': None,
                    'results': None,
                    'success': False,
                    'error': 'Invalid or missing group-by column for guided query.',
                    'answer_summary': '',
                    'warnings': [],
                    'trace': ['Guided route failed: invalid group-by column'],
                    'interpreted_intent': {'route': 'guided_group_by'},
                    'relevant_tables': [table_name],
                    'query_plan': {},
                    'repair_attempts': 0,
                    'needs_clarification': False,
                }
            q_group = self._quote_ident(group_by_column)
            sql_query = (
                f"SELECT {q_group} AS {self._quote_ident(group_by_column)}, {agg_expr} AS {metric_alias} "
                f"FROM {q_table}{where_clause} "
                f"GROUP BY {q_group} "
                f"ORDER BY {metric_alias} DESC "
                f"LIMIT {limit}"
            )
        else:
            return {
                'question': question,
                'sql_query': None,
                'results': None,
                'success': False,
                'error': 'Unsupported guided intent.',
                'answer_summary': '',
                'warnings': [],
                'trace': [f"Guided route failed: unsupported intent '{intent}'"],
                'interpreted_intent': {'route': f'guided_{intent}'},
                'relevant_tables': [table_name],
                'query_plan': {},
                'repair_attempts': 0,
                'needs_clarification': False,
            }

        try:
            results = self.files_db.execute_query(sql_query)
            if intent == 'aggregate' and results is not None and not results.empty:
                val = results.iloc[0, 0]
                answer_summary = f"{agg.upper()} result from '{table_name}': {val}."
            elif intent == 'group_by':
                answer_summary = f"Top {min(limit, len(results) if results is not None else 0)} grouped result rows from '{table_name}'."
            else:
                answer_summary = f"Guided query returned {len(results) if results is not None else 0} row(s)."

            self.add_to_history(question, {
                'sql_query': sql_query,
                'results': results,
                'success': True,
            })
            return {
                'question': question,
                'sql_query': sql_query,
                'results': results,
                'success': True,
                'error': None,
                'answer_summary': answer_summary,
                'warnings': [],
                'trace': [f"Guided route executed: guided_{intent}"],
                'interpreted_intent': {'route': f'guided_{intent}'},
                'relevant_tables': [table_name],
                'query_plan': {
                    'route': f'guided_{intent}',
                    'table': table_name,
                    'aggregation': agg,
                    'value_column': value_column,
                    'group_by_column': group_by_column,
                    'where_column': where_column,
                    'where_mode': where_mode,
                    'limit': limit,
                },
                'repair_attempts': 0,
                'needs_clarification': False,
            }
        except Exception as e:
            return {
                'question': question,
                'sql_query': sql_query,
                'results': None,
                'success': False,
                'error': str(e),
                'answer_summary': '',
                'warnings': [],
                'trace': [f"Guided route execution failed: {e}"],
                'interpreted_intent': {'route': f'guided_{intent}'},
                'relevant_tables': [table_name],
                'query_plan': {},
                'repair_attempts': 0,
                'needs_clarification': False,
            }

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str,
                  max_tokens: int = 800, temperature: float = 0.0) -> str:
        response = self.azure_client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _strip_fences(text: str, lang: str = "") -> str:
        text = re.sub(r"```" + lang + r"\s*", "", text, flags=re.IGNORECASE)
        return text.replace("```", "").strip()

    # ------------------------------------------------------------------
    # ── STEP 0: Clarification ─────────────────────────────────────────
    # ------------------------------------------------------------------

    # Known aggregation-hint words that suggest a value may be a subtotal / grouping row
    _GROUPING_ROW_HINTS = {
        "total", "subtotal", "grand total", "all", "sum",
        "overall", "aggregate", "combined", "group", "net",
    }

    def _get_schema_with_samples(self) -> str:
        """
        Compact schema block with distinct value samples for low-cardinality
        text columns.  Used only by the clarification check.

        Also flags columns where some values appear to be grouping / subtotal rows
        (values containing aggregation-hint words, or values that are a prefix of
        other values in the same column).  This lets the LLM warn the user when a
        table mixes detail rows with summary rows.
        """
        lines = []
        for tname, info in self.files_db.tables_info.items():
            lines.append(f"Table: {tname} [{info['row_count']} rows]")
            col_types = info.get('column_types', {})

            for col in info['columns']:
                dtype   = str(col_types.get(col, 'TEXT'))
                is_text = any(k in dtype.lower() for k in ('object', 'str', 'text'))

                if is_text:
                    try:
                        n_res = self.files_db.execute_query(
                            f'SELECT COUNT(DISTINCT TRIM("{col}")) as n FROM {tname}'
                        )
                        n_distinct = int(n_res['n'].iloc[0])

                        if n_distinct <= 30:
                            vdf  = self.files_db.execute_query(
                                f'SELECT DISTINCT TRIM("{col}") as v FROM {tname} '
                                f'WHERE "{col}" IS NOT NULL ORDER BY v LIMIT 20'
                            )
                            vals = [str(v) for v in vdf['v'].tolist()
                                    if str(v) not in ('nan', 'None', '')]

                            sample = ', '.join(f'"{v}"' for v in vals[:15])
                            col_line = f"  - {col} ({dtype}): {sample}"

                            # ── Grouping-row detection ─────────────────────
                            # Check 1: any value contains an aggregation-hint word
                            grouping_vals = [
                                v for v in vals
                                if any(
                                    hint in v.lower()
                                    for hint in self._GROUPING_ROW_HINTS
                                )
                            ]
                            # Check 2: any value is a prefix of another value
                            # (e.g. "Fixed Income" and "Fixed Income Securities")
                            prefix_vals = [
                                v for v in vals
                                if any(
                                    other.lower().startswith(v.lower() + " ")
                                    for other in vals
                                    if other != v and len(v) > 3
                                )
                            ]
                            suspect = list(dict.fromkeys(grouping_vals + prefix_vals))
                            if suspect:
                                col_line += (
                                    f'  ⚠️ possible grouping/subtotal rows: '
                                    f'{", ".join(f"{chr(34)}{v}{chr(34)}" for v in suspect[:3])}'
                                )

                            lines.append(col_line)
                        else:
                            vdf  = self.files_db.execute_query(
                                f'SELECT DISTINCT TRIM("{col}") as v FROM {tname} '
                                f'WHERE "{col}" IS NOT NULL LIMIT 3'
                            )
                            vals = [str(v) for v in vdf['v'].tolist()
                                    if str(v) not in ('nan', 'None', '')]
                            sample = ', '.join(f'"{v}"' for v in vals)
                            lines.append(
                                f"  - {col} ({dtype}): {sample}, "
                                f"... ({n_distinct} distinct values)"
                            )
                    except Exception:
                        lines.append(f"  - {col} ({dtype})")
                else:
                    lines.append(f"  - {col} ({dtype})")

            lines.append("")

        return "\n".join(lines)

    def _fetch_ranked_entity_options(
        self, table: str, column: str, search_term: str
    ) -> List[Tuple[str, str]]:
        """
        Find distinct values of `column` in `table` relevant to `search_term`,
        ranked by match quality. Returns a list of (value, label) tuples.

        Match tiers (lower = better rank):
          0 - exact match (case-insensitive)
          1 - value starts with term
          2 - term is a substring of value
          3 - any individual word of term is a substring of value

        No fallback to all-values — if nothing matches, returns [].
        This is the fix for the noise problem.
        """
        term        = search_term.strip()
        safe        = term.replace("'", "''")
        found: Dict[str, int] = {}   # value → best tier

        def _run(where: str, limit: int = 20) -> List[str]:
            try:
                df = self.files_db.execute_query(
                    f'SELECT DISTINCT TRIM("{column}") as v FROM {table} '
                    f"WHERE {where} AND \"{column}\" IS NOT NULL ORDER BY v LIMIT {limit}"
                )
                return [str(v) for v in df["v"].tolist()
                        if str(v) not in ("nan", "None", "")]
            except Exception:
                return []

        # Tier 0: exact (case-insensitive via SQLite LOWER)
        for v in _run(f"LOWER(TRIM(\"{column}\")) = LOWER('{safe}')"):
            found.setdefault(v, 0)

        # Tier 1: value starts with term
        for v in _run(f"LOWER(TRIM(\"{column}\")) LIKE LOWER('{safe}%')"):
            found.setdefault(v, 1)

        # Tier 2: term is a substring of value
        for v in _run(f"\"{column}\" LIKE '%{safe}%'"):
            found.setdefault(v, 2)

        # Tier 3: individual words (for multi-word terms like "Allianz Group")
        words = [w for w in re.split(r"\s+", term) if len(w) > 2]
        for word in words:
            safe_w = word.replace("'", "''")
            for v in _run(f"\"{column}\" LIKE '%{safe_w}%'"):
                found.setdefault(v, 3)

        if not found:
            return []

        # Sort by tier, then alphabetically within the same tier
        ranked = sorted(found.items(), key=lambda x: (x[1], x[0]))
        return [(v, "") for v, _ in ranked[:8]]   # return up to 8; caller will trim to 5

    def check_clarification_needed(
        self, question: str
    ) -> Optional[ClarificationRequest]:
        """
        Step 0 of the pipeline.

        v6 improvements over v5:
        - Schema-first constraint: options MUST reference actual values from the schema,
          not invented analyst bundles or conceptual groupings.
        - DB enrichment extended to non_trivial_assumption (not just fuzzy_entity/mixed).
        - Explicit anti-patterns listed in the prompt to prevent "liquidity proxy"-style options.
        - Grouping-row warnings surface in schema_with_samples and can flow into secondary_note.
        """
        schema_with_samples = self._get_schema_with_samples()

        system = """You are a data query clarification assistant.

Your task: decide if a user question needs clarification before SQL can be generated.

ONLY flag needs_clarification=true if the question contains a term that CANNOT be
deterministically resolved from the schema without making a non-obvious business assumption.

DO NOT flag needs_clarification if:
- The question uses exact column names or table names shown in the schema
- The question references a value that appears literally in the sample data
- The question is about a standard aggregation (count, sum, average, max, min)
- The question is exploratory ("show me", "list", "display", "how many rows")
- The intent is unambiguous even if the phrasing is informal

DO flag needs_clarification if:
- The question uses a business concept that could map to multiple categorical values
  (e.g. "fixed income exposure" when the schema has an asset_class column with several
  debt-related values; "equity exposure" when multiple equity categories exist)
- The question uses a geographic or organizational term with multiple valid interpretations
  (e.g. "Benelux" could mean the geographic region OR a legal entity)
- The question references an entity name not exactly present in the sample data
- Answering correctly requires selecting rows in a way that depends on non-obvious grouping

=== SCHEMA-FIRST RULE — READ CAREFULLY ===
ALL clarification options MUST be directly grounded in values that are explicitly shown
in the schema sample data below. This is a hard constraint.

DO NOT:
- Invent option labels like "other fixed-income-like assets", "liquidity proxy",
  "very broad exposure", "all asset classes as proxy", "including affiliates"
- Create conceptual bundles that are not represented by exact schema values
- Suggest combining categories unless those exact category values appear in the schema
- Use analyst-style interpretations or financial theory to generate options
- Add options "just to be complete" if they are not grounded in schema values

DO:
- List actual categorical values from the schema that match the concept
- Use the lookup field to signal which column and table contains the relevant values
- Let the system query the database to find real matching values
- When in doubt about which values qualify, ask a narrower question and let the user choose

If the schema sample does not contain enough information to list grounded options,
return clarification_options: [] and let the lookup field do the work.

=== GROUPING ROW WARNING ===
If the schema shows a column annotated with "⚠️ possible grouping/subtotal rows",
this means some values in that column may represent aggregated totals rather than
detail-level categories. If the user's question touches that column, add a brief note
in secondary_note such as:
"Note: some values in [column] may be subtotal or grouping rows. Filtering on those
may include already-aggregated data."

=== AMBIGUITY TYPES ===
- "fuzzy_entity"           : entity/filter value does not exactly match sample data
- "ambiguous_metric"       : metric term could map to multiple numeric columns
- "missing_filter_value"   : a filter dimension is unclear (which category values to include?)
- "non_trivial_assumption" : answering requires selecting from multiple categorical values
                             whose grouping is non-obvious (e.g. which asset_class values
                             count as "fixed income"?)
- "mixed"                  : question has BOTH a row/category ambiguity AND a metric/column ambiguity

=== OUTPUT FORMAT ===
Return ONLY valid JSON (no markdown, no explanation):
{
  "needs_clarification": true|false,
  "ambiguity_type": "fuzzy_entity"|"ambiguous_metric"|"missing_filter_value"|"non_trivial_assumption"|"mixed"|null,
  "ambiguous_term": "exact phrase from the question, or null",
  "clarification_question": "concise question naming the ambiguity — do not propose interpretations not in the schema",
  "clarification_reason": "brief internal reason",
  "clarification_options": ["option1", "option2", ...],
  "option_labels": ["Short label 1", "Short label 2", ...],
  "lookup": {"table": "...", "column": "..."} or null,
  "secondary_note": "grouping-row warning or other residual note, or null"
}

=== RULES FOR clarification_options ===

For "non_trivial_assumption" (e.g. "fixed income exposure", "equity exposure"):
  - Set lookup to the most relevant categorical column (e.g. asset_class)
  - The system will query the DB for real matching values — do NOT invent them
  - In clarification_options, list ONLY values that appear literally in the schema sample
  - If you cannot see enough values to fill the list, return [] — the DB lookup will fill it
  - option_labels: "" (empty) for plain schema values; brief label only if meaningful
  - Example for "fixed income exposure" when asset_class sample shows
    "Debt Securities", "Debt Funds", "Bonds", "Cash":
      options: ["Debt Securities", "Debt Funds"]   ← only what the schema literally shows
      option_labels: ["", ""]
      lookup: {"table": "investments", "column": "asset_class"}
      clarification_question: "When you say 'fixed income', which asset_class values should I include?"
  - DO NOT add: "other fixed-income-like assets" — not in schema

For "fuzzy_entity":
  - List exact entity values from the sample data, ranked by closeness to the term
  - If the term is also a geographic concept (e.g. "Benelux"), include the geographic
    interpretation as the first option with a clear label
  - lookup: the entity column so the DB can find ranked matches
  - Example for "Allianz Group":
      options: ["Allianz SE", "Allianz Benelux"]   ← from schema sample
      option_labels: ["", ""]
      lookup: {"table": "...", "column": "company_name"}

For "ambiguous_metric":
  - options: actual column names from the schema
  - option_labels: plain-language description of what each column measures

For "mixed":
  - clarification_question must name BOTH dimensions clearly
  - options must be grounded in actual schema values — no invented bundles
  - lookup: the categorical column that needs to be filtered
  - secondary_note: mention the metric ambiguity if it remains after the row filter is resolved

For "missing_filter_value":
  - options: distinct values of the relevant categorical column from the schema
  - lookup: that column so the DB can find all real values

Maximum 5 options total. Prefer fewer, higher-quality, schema-grounded options.
Return [] for options rather than inventing values not shown in the schema."""

        user = (
            f"Schema (with sample values for text columns):\n"
            f"{schema_with_samples}\n\n"
            f"User question: {question}"
        )

        try:
            raw    = self._call_llm(system, user, max_tokens=600)
            raw    = self._strip_fences(raw, "json")
            parsed = json.loads(raw)
        except Exception:
            return None

        if not parsed.get("needs_clarification"):
            return None

        # ── Post-process options ───────────────────────────────────────────
        llm_options: List[str] = parsed.get("clarification_options") or []
        llm_labels:  List[str] = parsed.get("option_labels") or []

        while len(llm_labels) < len(llm_options):
            llm_labels.append("")

        ambiguity_type = parsed.get("ambiguity_type", "non_trivial_assumption")
        lookup         = parsed.get("lookup") or {}
        table          = lookup.get("table", "")
        column         = lookup.get("column", "")
        term           = parsed.get("ambiguous_term") or ""

        # DB enrichment applies to ALL types that provide a lookup, not just fuzzy_entity.
        # This is the key fix: non_trivial_assumption now gets real DB values too.
        if table and column and term and table in self.files_db.tables_info:
            db_pairs = self._fetch_ranked_entity_options(table, column, term)
            db_vals  = [v for v, _ in db_pairs]

            if ambiguity_type == "fuzzy_entity":
                # Keep labeled conceptual options first, replace unlabeled ones with DB values.
                conceptual_opts   = [(o, l) for o, l in zip(llm_options, llm_labels) if l]
                non_conceptual_db = [(v, "") for v in db_vals
                                      if v not in {o for o, _ in conceptual_opts}]
                merged       = conceptual_opts + non_conceptual_db
                final_opts   = [o for o, _ in merged[:5]]
                final_labels = [l for _, l in merged[:5]]

            elif ambiguity_type in ("non_trivial_assumption", "missing_filter_value"):
                # Replace all LLM options with ranked real DB values.
                # LLM options that happen to exactly match real DB values are preserved
                # (they are schema-grounded); purely invented ones are dropped.
                db_val_set = set(db_vals)
                # Keep any LLM option that is an exact DB value
                grounded = [(o, l) for o, l in zip(llm_options, llm_labels) if o in db_val_set]
                # Add remaining DB values not already in grounded list
                grounded_vals = {o for o, _ in grounded}
                extra = [(v, "") for v in db_vals if v not in grounded_vals]
                merged       = grounded + extra
                final_opts   = [o for o, _ in merged[:5]]
                final_labels = [l for _, l in merged[:5]]

            elif ambiguity_type == "mixed":
                # Keep LLM structured options, but verify any that look like schema values.
                # Append remaining DB values if there's room.
                existing_vals = set(llm_options)
                extra = [(v, "") for v in db_vals if v not in existing_vals]
                merged_opts   = llm_options + [v for v, _ in extra]
                merged_labels = llm_labels  + ["" for _ in extra]
                final_opts    = merged_opts[:5]
                final_labels  = merged_labels[:5]

            else:
                final_opts   = llm_options[:5]
                final_labels = llm_labels[:5]
        else:
            final_opts   = llm_options[:5]
            final_labels = llm_labels[:5]

        while len(final_labels) < len(final_opts):
            final_labels.append("")

        return ClarificationRequest(
            ambiguity_type=ambiguity_type,
            ambiguous_term=term,
            clarification_question=(
                parsed.get("clarification_question")
                or "Could you clarify what you mean?"
            ),
            clarification_reason=parsed.get("clarification_reason") or "",
            clarification_options=final_opts,
            option_labels=final_labels,
            secondary_note=parsed.get("secondary_note") or "",
        )

    # ------------------------------------------------------------------
    # Step 1 — Intent parsing
    # ------------------------------------------------------------------

    def parse_intent(self, question: str) -> ParsedIntent:
        system = (
            "You are a data query intent parser.\n"
            "Return ONLY a valid JSON object with exactly these keys:\n"
            '{\n'
            '  "action": one of ["aggregate","filter","list","count","compare","lookup"],\n'
            '  "entities": [list of key nouns/concepts, lowercase],\n'
            '  "filters": {column hints to value hints},\n'
            '  "aggregation": one of ["sum","count","avg","max","min","none"],\n'
            '  "group_by_hint": string or null,\n'
            '  "sort_hint": string or null,\n'
            '  "sort_order": "asc" or "desc",\n'
            '  "limit": integer or null\n'
            "}\nNo explanation. No markdown. Only the JSON object."
        )
        try:
            raw = self._call_llm(system, f"Parse this data query:\n{question}",
                                 max_tokens=350)
            raw = self._strip_fences(raw, "json")
            d   = json.loads(raw)
        except Exception:
            d = {
                "action": "list", "entities": [], "filters": {},
                "aggregation": "none", "group_by_hint": None,
                "sort_hint": None, "sort_order": "asc", "limit": None,
            }
        return ParsedIntent(
            action=d.get("action", "list"),
            entities=[str(e).lower() for e in d.get("entities", [])],
            filters=d.get("filters", {}),
            aggregation=d.get("aggregation", "none"),
            group_by_hint=d.get("group_by_hint"),
            sort_hint=d.get("sort_hint"),
            sort_order=d.get("sort_order", "asc"),
            limit=d.get("limit"),
            raw=d,
        )

    # ------------------------------------------------------------------
    # Step 2 — Relevant schema selection (deterministic keyword scoring)
    # ------------------------------------------------------------------

    def select_relevant_schema(self, intent: ParsedIntent) -> Dict[str, List[Dict]]:
        if not self.schema_info:
            return {}
        if len(self.schema_info) == 1:
            return dict(self.schema_info)

        tokens: set = set(intent.entities)
        for key in intent.filters:
            tokens.update(re.split(r'[\s_]', key.lower()))
        for hint in (intent.group_by_hint, intent.sort_hint):
            if hint:
                tokens.update(re.split(r'[\s_]', hint.lower()))
        tokens = {t for t in tokens if len(t) > 1}

        scores: Dict[str, int] = {}
        for tname, columns in self.schema_info.items():
            score     = 0
            ttokens   = set(re.split(r'[_\s]', tname.lower()))
            score    += len(tokens & ttokens) * 3
            col_names = [c['name'].lower() for c in columns]
            for tok in tokens:
                for col in col_names:
                    if tok in col or col in tok:
                        score += 1
            scores[tname] = score

        max_score = max(scores.values(), default=0)
        if max_score == 0:
            return dict(self.schema_info)

        threshold = max(1, max_score * 0.3)
        relevant  = {t: self.schema_info[t] for t, s in scores.items() if s >= threshold}
        if not relevant:
            best     = max(scores, key=scores.get)
            relevant = {best: self.schema_info[best]}
        return relevant

    # ------------------------------------------------------------------
    # Schema formatter
    # ------------------------------------------------------------------

    def _format_schema_for_prompt(
        self, schema: Dict[str, List[Dict]], include_samples: bool = True
    ) -> str:
        parts = []
        for tname, columns in schema.items():
            info      = self.files_db.tables_info.get(tname, {})
            col_lines = "\n".join(f"  - {c['name']} ({c['type']})" for c in columns)
            block     = (
                f"Table: {tname}  [source: {info.get('source_file','?')}, "
                f"rows: {info.get('row_count','?')}]\n{col_lines}"
            )
            if include_samples:
                try:
                    sample_df = self.files_db.execute_query(
                        f"SELECT * FROM {tname} LIMIT 3"
                    )
                    block += f"\nSample data:\n{sample_df.to_string(index=False)}"
                except Exception:
                    pass
            parts.append(block)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Step 3 — Query planning
    # ------------------------------------------------------------------

    def build_query_plan(
        self, question: str, intent: ParsedIntent, schema: Dict
    ) -> QueryPlan:
        schema_text = self._format_schema_for_prompt(schema, include_samples=True)
        system = (
            "You are a SQL query planner for SQLite.\n"
            "Return ONLY a valid JSON object:\n"
            "{\n"
            '  "tables": [table names to use],\n'
            '  "columns": [columns to SELECT — use * only as last resort],\n'
            '  "filters": [{"column":"...","operator":"=|!=|>|<|>=|<=|LIKE|IS NULL","value":"..."}],\n'
            '  "aggregation": {"function":"SUM|COUNT|AVG|MAX|MIN","column":"..."} or null,\n'
            '  "group_by": [column names],\n'
            '  "order_by": {"column":"...","direction":"ASC|DESC"} or null,\n'
            '  "limit": integer or null,\n'
            '  "joins": [{"left_table":"...","right_table":"...","left_col":"...","right_col":"..."}],\n'
            '  "notes": "brief notes"\n'
            "}\nNo explanation. No markdown. Only the JSON."
        )
        user = (
            f"Schema:\n{schema_text}\n\nUser question: {question}\n\n"
            f"Parsed intent: action={intent.action}, aggregation={intent.aggregation}, "
            f"entities={intent.entities}\n\nCreate the query plan:"
        )
        try:
            raw = self._call_llm(system, user, max_tokens=600)
            raw = self._strip_fences(raw, "json")
            d   = json.loads(raw)
        except Exception:
            d = {
                "tables": list(schema.keys())[:1], "columns": ["*"],
                "filters": [], "aggregation": None, "group_by": [],
                "order_by": None, "limit": 100, "joins": [],
                "notes": "fallback plan",
            }
        return QueryPlan(
            tables=d.get("tables", []),
            columns=d.get("columns", ["*"]),
            filters=d.get("filters", []),
            aggregation=d.get("aggregation"),
            group_by=d.get("group_by", []),
            order_by=d.get("order_by"),
            limit=d.get("limit"),
            joins=d.get("joins", []),
            notes=d.get("notes", ""),
        )

    # ------------------------------------------------------------------
    # Step 4 — SQL generation from plan
    # ------------------------------------------------------------------

    def generate_sql_from_plan(
        self, question: str, intent: ParsedIntent,
        plan: QueryPlan, schema: Dict
    ) -> str:
        schema_text = self._format_schema_for_prompt(schema, include_samples=False)
        plan_json   = json.dumps({
            "tables": plan.tables, "columns": plan.columns,
            "filters": plan.filters, "aggregation": plan.aggregation,
            "group_by": plan.group_by, "order_by": plan.order_by,
            "limit": plan.limit, "joins": plan.joins,
        }, indent=2)
        system = (
            "You are a SQLite SQL expert.\n"
            "Generate ONLY a valid SQLite SELECT (or WITH) query.\n"
            "No explanations. No markdown. No comments. Just raw SQL."
        )
        user = (
            f"Schema:\n{schema_text}\n\n"
            f"{self.get_conversation_context()}\n"
            f"User question: {question}\n\nQuery plan:\n{plan_json}\n\n"
            "Rules:\n"
            "- Use exact table and column names from the schema\n"
            "- SQLite syntax only (no ILIKE, no ::cast)\n"
            "- Use TRIM() for text columns that may have extra spaces\n"
            "- Use LIKE '%value%' for partial matches\n"
            "- Only SELECT or WITH queries are allowed\n\n"
            "Generate the SQL:"
        )
        raw = self._call_llm(system, user, max_tokens=4000)
        return self._strip_fences(raw, "sql")

    # ------------------------------------------------------------------
    # Step 5 — Deterministic SQL validation
    # ------------------------------------------------------------------

    def validate_sql(
        self, sql: str, plan: QueryPlan
    ) -> Tuple[bool, str, List[str]]:
        warn    = []
        cleaned = sql.strip().rstrip(";").strip()
        upper   = cleaned.upper().lstrip()

        if not (upper.startswith("SELECT") or upper.startswith("WITH")):
            return False, cleaned, ["SQL does not start with SELECT or WITH — blocked."]

        for kw in _DANGEROUS_KEYWORDS:
            # Allow SQLite string REPLACE() function in read-only SELECT queries.
            # Still block mutating forms such as REPLACE INTO.
            if kw == "REPLACE":
                has_replace_into = bool(re.search(r"\bREPLACE\s+INTO\b", cleaned, re.IGNORECASE))
                has_replace_function = bool(re.search(r"\bREPLACE\s*\(", cleaned, re.IGNORECASE))
                if has_replace_into:
                    return False, cleaned, ["Dangerous keyword 'REPLACE INTO' found — blocked."]
                if has_replace_function:
                    continue

            if re.search(r'\b' + kw + r'\b', cleaned, re.IGNORECASE):
                return False, cleaned, [f"Dangerous keyword '{kw}' found — blocked."]

        if ";" in cleaned:
            cleaned = cleaned.split(";")[0].strip()
            warn.append("Multiple SQL statements detected; only the first was kept.")

        has_limit  = bool(re.search(r'\bLIMIT\b', cleaned, re.IGNORECASE))
        has_agg    = plan.aggregation is not None or bool(plan.group_by)
        has_agg_fn = bool(re.search(
            r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', cleaned, re.IGNORECASE
        ))
        if not has_limit and not has_agg and not has_agg_fn:
            cleaned += " LIMIT 1000"
            warn.append("No LIMIT on non-aggregate query — LIMIT 1000 applied automatically.")

        return True, cleaned, warn

    # ------------------------------------------------------------------
    # Step 6 — SQL repair
    # ------------------------------------------------------------------

    def _repair_sql(
        self, question: str, intent: ParsedIntent, plan: QueryPlan,
        failed_sql: str, error_msg: str, schema: Dict
    ) -> str:
        schema_text = self._format_schema_for_prompt(schema, include_samples=False)
        plan_json   = json.dumps({
            "tables": plan.tables, "columns": plan.columns,
            "filters": plan.filters, "aggregation": plan.aggregation,
            "group_by": plan.group_by, "order_by": plan.order_by,
            "limit": plan.limit, "joins": plan.joins,
        }, indent=2)
        system = (
            "You are a SQLite SQL repair expert.\n"
            "Fix the broken SQL query based on the exact database error provided.\n"
            "Return ONLY the corrected SQL. No explanation. No markdown."
        )
        user = (
            f"Schema:\n{schema_text}\n\n"
            f"Original question: {question}\n"
            f"Intent: action={intent.action}, aggregation={intent.aggregation}\n\n"
            f"Query plan:\n{plan_json}\n\n"
            f"Failed SQL:\n{failed_sql}\n\n"
            f"Exact database error:\n{error_msg}\n\n"
            "Return the corrected SQLite SELECT query:"
        )
        raw = self._call_llm(system, user, max_tokens=4000)
        return self._strip_fences(raw, "sql")

    # ------------------------------------------------------------------
    # Step 7a — Result analysis  (pure Python, no LLM call)
    # ------------------------------------------------------------------

    # Column name tokens that suggest an entity/dimension (not a metric).
    _ENTITY_COLUMN_HINTS = {
        "name", "company", "entity", "group", "region", "country",
        "division", "segment", "unit", "department", "category",
        "product", "market", "brand", "client", "customer",
    }

    def _analyze_result(
        self, results: pd.DataFrame, plan: QueryPlan, sql: str
    ) -> ResultFlags:
        """
        Inspect the execution result for suspicious patterns — purely in Python.

        Checks performed (in order):
          1. Empty result set.
          2. Aggregate returned NULL or 0.
          3. Exact-match (=) filter on a text/entity-like column → flag as
             possible_exact_match_miss and run a LIKE diagnostic query to
             find similar values.
        """
        flags = ResultFlags()

        # ── 1. Empty result set ────────────────────────────────────────────
        if results is None or results.empty:
            flags.empty_result = True

        # ── 2. Aggregate returned 0 or NULL ───────────────────────────────
        if (
            not flags.empty_result
            and plan.aggregation is not None
            and len(results) == 1
            and len(results.columns) >= 1
        ):
            val = results.iloc[0, 0]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                flags.suspicious_zero_result = True
            elif isinstance(val, (int, float)) and float(val) == 0.0:
                flags.suspicious_zero_result = True

        # ── 3. Exact-match filter on entity-like text column ───────────────
        # Only inspect plan filters with operator "=" on columns whose name
        # suggests they hold entity/dimension values.
        exact_text_filters = [
            f for f in (plan.filters or [])
            if str(f.get("operator", "")).strip() == "="
            and any(
                hint in str(f.get("column", "")).lower()
                for hint in self._ENTITY_COLUMN_HINTS
            )
        ]

        if exact_text_filters and (flags.empty_result or flags.suspicious_zero_result):
            # Pick the first entity-like filter to investigate.
            suspect = exact_text_filters[0]
            col     = str(suspect.get("column", ""))
            val     = str(suspect.get("value", ""))
            flags.filter_column = col
            flags.filter_value  = val

            if col and val:
                flags.possible_exact_match_miss = True

                # Run a lightweight LIKE query to surface similar real values.
                similar: List[str] = []
                # Try each relevant table.
                for tname in plan.tables:
                    if tname not in self.files_db.tables_info:
                        continue
                    table_cols = [
                        c['name'] for c in self.schema_info.get(tname, [])
                    ]
                    if col not in table_cols:
                        continue
                    safe_val = val.replace("'", "''")
                    try:
                        df_like = self.files_db.execute_query(
                            f'SELECT DISTINCT TRIM("{col}") as v FROM {tname} '
                            f"WHERE \"{col}\" LIKE '%{safe_val}%' "
                            f"ORDER BY v LIMIT 10"
                        )
                        similar = [
                            str(v) for v in df_like["v"].tolist()
                            if str(v) not in ("nan", "None", "")
                            and str(v).strip() != val.strip()
                        ]
                    except Exception:
                        pass
                    if similar:
                        break  # found results from this table; stop

                if similar:
                    flags.entity_match_uncertain = True
                    flags.similar_values = similar[:5]

        return flags

    # ------------------------------------------------------------------
    # Step 7b — Answer summary  (calibrated to ResultFlags)
    # ------------------------------------------------------------------

    def _generate_answer_summary(
        self, question: str, sql: str,
        results: pd.DataFrame, flags: ResultFlags
    ) -> str:
        """
        Generate a natural-language summary.
        When ResultFlags indicate a suspicious outcome, the system prompt
        explicitly forbids strong causal claims and requires careful phrasing.
        """
        # ── Empty result ───────────────────────────────────────────────────
        if flags.empty_result:
            if flags.possible_exact_match_miss:
                base = (
                    f"No rows were returned for the current filter "
                    f'("{flags.filter_column}" = "{flags.filter_value}"). '
                )
                if flags.similar_values:
                    alts = ", ".join(f'"{v}"' for v in flags.similar_values)
                    base += (
                        f"Similar values exist in the data ({alts}), so the filter "
                        f"may not have matched the intended entity."
                    )
                else:
                    base += (
                        "This may indicate that no data exists for this entity, "
                        "or that the name was not matched exactly."
                    )
                return base
            return "No rows were returned for this query."

        # ── Aggregate returned 0 or NULL ───────────────────────────────────
        if flags.suspicious_zero_result:
            val_str = str(results.iloc[0, 0]) if not results.empty else "0"
            base = (
                f"The query returned a result of {val_str} for the specified filter. "
            )
            if flags.possible_exact_match_miss:
                base += (
                    f'This may reflect a true zero, or the filter on '
                    f'"{flags.filter_column}" = "{flags.filter_value}" '
                    f"may not have matched the intended entity. "
                )
                if flags.similar_values:
                    alts = ", ".join(f'"{v}"' for v in flags.similar_values)
                    base += f"Similar values in the data include: {alts}."
            else:
                base += (
                    "This may indicate a true zero value, or that the matched rows "
                    "contain null / zero entries for the requested field."
                )
            return base

        # ── Normal result: use LLM summary but with care instructions ──────
        sample  = results.head(10).to_string(index=False)
        is_suspicious = flags.possible_exact_match_miss or flags.entity_match_uncertain

        if is_suspicious:
            system = (
                "You are a careful data analyst assistant. "
                "Write a concise (2–3 sentences) plain-language answer. "
                "IMPORTANT RULES:\n"
                "- Do NOT make strong causal claims such as 'there was no activity' "
                "or 'this means nothing was recorded'.\n"
                "- If the result is zero or small, phrase it cautiously: "
                "'the query returned X', not 'X was the actual total'.\n"
                "- Do NOT interpret absence of data as a business fact.\n"
                "No markdown, no bullet points, no preamble."
            )
        else:
            system = (
                "You are a data analyst assistant. "
                "Write a concise (2–3 sentences) plain-language answer that directly "
                "addresses the user's question using the query results. "
                "No markdown, no bullet points, no preamble."
            )

        user = (
            f"Question: {question}\n\nSQL:\n{sql}\n\n"
            f"Result ({len(results)} rows total, showing up to 10):\n{sample}\n\nAnswer:"
        )
        try:
            return self._call_llm(system, user, max_tokens=250, temperature=0.2)
        except Exception:
            return f"The query returned {len(results)} row(s)."

    # ------------------------------------------------------------------
    # Steps 1–7: run_query
    # ------------------------------------------------------------------

    def run_query(self, question: str) -> QueryResponse:
        """Full pipeline — called only after clarification check returns None."""
        trace: List[str] = []
        warn:  List[str] = []

        # Step 1
        trace.append("Step 1: Parsing intent")
        try:
            intent = self.parse_intent(question)
            trace.append(f"  → action={intent.action}, agg={intent.aggregation}, entities={intent.entities}")
        except Exception as e:
            trace.append(f"  → fallback ({e})")
            intent = ParsedIntent(
                action="list",
                entities=[w for w in question.lower().split() if len(w) > 2][:10],
                filters={}, aggregation="none",
                group_by_hint=None, sort_hint=None, sort_order="asc", limit=None,
            )

        # Step 2
        trace.append("Step 2: Selecting relevant schema")
        try:
            relevant_schema = self.select_relevant_schema(intent)
            trace.append(f"  → tables: {list(relevant_schema.keys())}")
        except Exception as e:
            trace.append(f"  → fallback ({e})")
            relevant_schema = dict(self.schema_info)
            warn.append("Schema selection failed; full schema used.")

        # Step 3
        trace.append("Step 3: Building query plan")
        try:
            plan = self.build_query_plan(question, intent, relevant_schema)
            trace.append(f"  → tables={plan.tables}, agg={plan.aggregation}, group_by={plan.group_by}")
        except Exception as e:
            trace.append(f"  → fallback ({e})")
            plan = QueryPlan(
                tables=list(relevant_schema.keys())[:1], columns=["*"],
                filters=[], aggregation=None, group_by=[], order_by=None,
                limit=100, joins=[], notes="minimal fallback",
            )
            warn.append("Query planning failed; minimal fallback used.")

        # Step 4
        trace.append("Step 4: Generating SQL from plan")
        sql: Optional[str] = None
        try:
            sql = self.generate_sql_from_plan(question, intent, plan, relevant_schema)
            trace.append(f"  → {len(sql)} chars")
        except Exception as e:
            msg = f"SQL generation failed: {e}"
            trace.append(f"  → {msg}")
            return QueryResponse(
                question=question, interpreted_intent=intent.raw,
                relevant_tables=list(relevant_schema.keys()), query_plan=asdict(plan),
                sql_query=None, results=None, answer_summary="",
                warnings=warn, trace=trace, success=False, error=msg,
            )

        # Step 5
        trace.append("Step 5: Validating SQL")
        is_valid, sql, val_warn = self.validate_sql(sql, plan)
        warn.extend(val_warn)
        if not is_valid:
            trace.append(f"  → blocked: {val_warn}")
            return QueryResponse(
                question=question, interpreted_intent=intent.raw,
                relevant_tables=list(relevant_schema.keys()), query_plan=asdict(plan),
                sql_query=sql, results=None, answer_summary="",
                warnings=warn, trace=trace, success=False,
                error=val_warn[0] if val_warn else "SQL validation failed.",
            )
        trace.append(f"  → valid{': ' + '; '.join(val_warn) if val_warn else ''}")

        # Step 6: execute + repair
        trace.append("Step 6: Executing SQL (up to 3 attempts)")
        results:        Optional[pd.DataFrame] = None
        repair_attempts = 0
        last_error:     Optional[str] = None

        for attempt in range(3):
            try:
                results = self.files_db.execute_query(sql)
                trace.append(f"  → succeeded attempt {attempt + 1} ({len(results)} rows)")
                break
            except Exception as e:
                last_error = str(e)
                trace.append(f"  → failed attempt {attempt + 1}: {last_error}")
                if attempt < 2:
                    trace.append(f"  → repair {attempt + 1}/2 …")
                    try:
                        repaired = self._repair_sql(
                            question, intent, plan, sql, last_error, relevant_schema
                        )
                        ok, repaired, rep_warn = self.validate_sql(repaired, plan)
                        warn.extend(rep_warn)
                        if ok:
                            sql             = repaired
                            repair_attempts += 1
                            trace.append("  → repaired SQL accepted")
                        else:
                            trace.append("  → repaired SQL failed validation; stopping")
                            break
                    except Exception as rep_e:
                        trace.append(f"  → repair call failed: {rep_e}")
                        break

        if results is None:
            return QueryResponse(
                question=question, interpreted_intent=intent.raw,
                relevant_tables=list(relevant_schema.keys()), query_plan=asdict(plan),
                sql_query=sql, results=None, answer_summary="",
                warnings=warn, trace=trace, success=False,
                error=last_error, repair_attempts=repair_attempts,
            )

        # Step 7a: Result analysis (pure Python — no LLM call)
        trace.append("Step 7a: Analysing result for suspicious patterns")
        try:
            flags = self._analyze_result(results, plan, sql)
            trace.append(
                f"  → empty={flags.empty_result}, "
                f"zero={flags.suspicious_zero_result}, "
                f"match_miss={flags.possible_exact_match_miss}, "
                f"uncertain={flags.entity_match_uncertain}"
            )
            # Emit structured warnings from flags
            if flags.empty_result:
                if flags.possible_exact_match_miss:
                    msg = (
                        f'No rows matched the exact filter '
                        f'("{flags.filter_column}" = "{flags.filter_value}"). '
                    )
                    if flags.similar_values:
                        alts = ", ".join(f'"{v}"' for v in flags.similar_values)
                        msg += f"Similar values found in the data: {alts}."
                    else:
                        msg += "Consider checking the exact spelling of the entity name."
                    warn.append(msg)
                else:
                    warn.append("The query returned no rows for the current filter.")

            if flags.suspicious_zero_result:
                msg = (
                    "The aggregate result is 0 or null. "
                    "This may reflect a true zero, or the filter may not have "
                    "matched the intended rows."
                )
                if flags.possible_exact_match_miss:
                    msg = (
                        f'The result is 0 or null, and the filter used an exact match '
                        f'on "{flags.filter_column}" = "{flags.filter_value}". '
                        "This may not capture the intended entity."
                    )
                    if flags.similar_values:
                        alts = ", ".join(f'"{v}"' for v in flags.similar_values)
                        msg += f" Similar values in the data include: {alts}."
                warn.append(msg)

            if flags.entity_match_uncertain and not flags.suspicious_zero_result:
                # Non-zero result but entity matching is uncertain
                alts = ", ".join(f'"{v}"' for v in flags.similar_values)
                warn.append(
                    f'The result is based on an exact match for '
                    f'"{flags.filter_value}" on column "{flags.filter_column}". '
                    f"Other similar values exist ({alts}); verify this is the intended entity."
                )
        except Exception as e:
            flags = ResultFlags()
            trace.append(f"  → analysis failed ({e}); continuing without flags")

        # Step 7b: Answer summary (calibrated to ResultFlags)
        trace.append("Step 7b: Generating answer summary")
        try:
            answer_summary = self._generate_answer_summary(question, sql, results, flags)
            trace.append("  → done")
        except Exception as e:
            answer_summary = f"Query returned {len(results)} row(s)."
            trace.append(f"  → fallback ({e})")

        self.add_to_history(question, {
            'sql_query': sql, 'results': results, 'success': True
        })

        return QueryResponse(
            question=question, interpreted_intent=intent.raw,
            relevant_tables=list(relevant_schema.keys()), query_plan=asdict(plan),
            sql_query=sql, results=results, answer_summary=answer_summary,
            warnings=warn, trace=trace, success=True, repair_attempts=repair_attempts,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute_query_with_explanation(self, user_question: str) -> Dict:
        """
        Step 0 — check_clarification_needed:
          If ambiguous → return clarification dict immediately (no SQL generated).
        Steps 1–7 — run_query:
          If clear → run full pipeline and return result dict.

        Always returns a plain dict for backward compatibility with app.py.
        """
        # ── Step 0: Clarification check ──────────────────────────────────
        try:
            clarification = self.check_clarification_needed(user_question)
        except Exception:
            clarification = None  # on failure, proceed to query

        if clarification is not None:
            return {
                'question':               user_question,
                'success':                False,
                'needs_clarification':    True,
                'clarification_question': clarification.clarification_question,
                'clarification_reason':   clarification.clarification_reason,
                'clarification_options':  clarification.clarification_options,
                'option_labels':          clarification.option_labels,
                'secondary_note':         clarification.secondary_note,
                'sql_query':              None,
                'results':                None,
                'warnings':               [],
                'answer_summary':         '',
                'trace': [
                    f"Clarification requested ({clarification.ambiguity_type}): "
                    f"'{clarification.ambiguous_term}'"
                ],
                'error':              None,
                'repair_attempts':    0,
                'interpreted_intent': {},
                'relevant_tables':    [],
                'query_plan':         {},
            }

        # ── Steps 1–7: Full pipeline ──────────────────────────────────────
        r = self.run_query(user_question)
        return {
            'question':            r.question,
            'sql_query':           r.sql_query,
            'results':             r.results,
            'success':             r.success,
            'error':               r.error,
            'answer_summary':      r.answer_summary,
            'warnings':            r.warnings,
            'trace':               r.trace,
            'interpreted_intent':  r.interpreted_intent,
            'relevant_tables':     r.relevant_tables,
            'query_plan':          r.query_plan,
            'repair_attempts':     r.repair_attempts,
            'needs_clarification': False,
        }

    # ------------------------------------------------------------------
    # Legacy helpers
    # ------------------------------------------------------------------

    def generate_schema_description(self) -> str:
        return self._format_schema_for_prompt(self.schema_info, include_samples=False)

    def get_sample_data_summary(self) -> str:
        return self._format_schema_for_prompt(self.schema_info, include_samples=True)
