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
from datetime import date, datetime
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, cast
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from sqlalchemy import create_engine
from flat_file_builder import (
    build_embedded_schema_frame,
    build_excel_schema_package,
    to_multisheet_excel_bytes,
)
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
    ambiguity_type: str  # ambiguous_metric | fuzzy_entity | missing_filter_value | non_trivial_assumption | mixed
    ambiguous_term: str  # exact phrase in the question that triggered this
    clarification_question: str
    clarification_reason: str
    clarification_options: List[str]
    option_labels: List[str] = field(default_factory=list)  # parallel to options; short interpretation labels
    secondary_note: str = ""  # follow-up note about a residual ambiguity after the main one is resolved


@dataclass
class ParsedIntent:
    """Structured representation of what the user is asking."""
    action: str  # aggregate | filter | list | count | compare | lookup
    entities: List[str]
    filters: Dict[str, Any]
    aggregation: str  # sum | count | avg | max | min | none
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
    # Scope values the result is expressed in, e.g. {"unit": ["EUR mn"]}.
    result_scope: Dict[str, List[str]] = field(default_factory=dict)
    # Real values to offer when a filter matched nothing, e.g.
    # {"column": "company_name", "value": "Allianz Group",
    #  "alternatives": ["Allianz SE", "Allianz Benelux"]}.
    value_suggestions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultFlags:
    """
    Structured flags from post-execution result analysis.
    All fields are False / empty by default (non-suspicious result).
    """
    empty_result: bool = False  # result set has zero rows
    suspicious_zero_result: bool = False  # aggregate returned 0 or null
    possible_exact_match_miss: bool = False  # exact-match filter on text col + zero/empty
    entity_match_uncertain: bool = False  # similar values found in DB that were not matched
    similar_values: List[str] = field(default_factory=list)  # LIKE-found alternatives
    filter_column: str = ""  # the column that triggered the concern
    filter_value: str = ""  # the value that was searched


# ============================================================================
# 3. SQL safety constants
# ============================================================================

def _quoted_list(values: List[str]) -> str:
    """Render values as a comma-separated, double-quoted list for user-facing text."""
    return ", ".join(f'"{v}"' for v in values)


def _guided_result(
        question: str,
        route: str,
        *,
        success: bool,
        sql_query: Optional[str] = None,
        results: Optional[pd.DataFrame] = None,
        error: Optional[str] = None,
        answer_summary: str = '',
        warnings: Optional[List[str]] = None,
        trace: Optional[List[str]] = None,
        relevant_tables: Optional[List[str]] = None,
        query_plan: Optional[Dict] = None,
) -> Dict:
    """Build the result dict every deterministic guided route returns.

    Guided routes never generate SQL with an LLM and never ask for
    clarification, so those fields are fixed.
    """
    return {
        'question': question,
        'sql_query': sql_query,
        'results': results,
        'success': success,
        'error': error,
        'answer_summary': answer_summary,
        'warnings': warnings or [],
        'trace': trace or [],
        'interpreted_intent': {'route': route},
        'relevant_tables': relevant_tables or [],
        'query_plan': query_plan or {},
        'repair_attempts': 0,
        'needs_clarification': False,
    }


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
    try:
        from azure.identity import ClientSecretCredential, DefaultAzureCredential
    except ImportError as e:
        raise ImportError(
            "Missing Azure identity dependency. Install azure-identity in the "
            "Python environment used to run the app."
        ) from e

    client_id = os.getenv("AZURE_CLIENT_ID")
    tenant_id = os.getenv("AZURE_TENANT_ID")
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
        from openai import AzureOpenAI
        from azure.identity import get_bearer_token_provider
    except ImportError as e:
        raise ImportError(
            "Missing Azure/OpenAI dependencies. Install openai and "
            "azure-identity in the Python environment used to run the app."
        ) from e

    token_provider = get_bearer_token_provider(
        _build_credential(),
        "https://cognitiveservices.azure.com/.default",
    )
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
    )
    return client, AzureConfig(endpoint, deployment, api_version)


# ============================================================================
# 5. DATABASE MANAGER  (unchanged)
# ============================================================================

class FilesDatabaseManager:
    def __init__(self):
        self.engine = None
        self.connection = None
        self.tables_info: Dict = {}
        self.loaded_files: List = []

    def load_file(self, file_path: str, sheet_names: List[str] = None,
                  table_name: str = None) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.xlsx', '.xls']:
            return self.load_excel_file(file_path, sheet_names)
        if ext == '.csv':
            return self.load_csv_file(file_path, table_name)
        print(f"Unsupported file format: {ext}")
        return False

    def ensure_connection(self) -> None:
        """Create the in-memory SQLite engine/connection on first use."""
        if self.engine is None:
            self.engine = create_engine(
                'sqlite:///:memory:',
                connect_args={'check_same_thread': False}
            )
            self.connection = self.engine.connect()

    def register_table(self, table_name: str, df: pd.DataFrame, **source_info) -> None:
        """Write a dataframe to SQLite and record its schema in tables_info."""
        df.to_sql(table_name, self.connection, if_exists='replace', index=False)
        self.tables_info[table_name] = {
            **source_info,
            'columns': list(df.columns),
            'row_count': len(df),
            'column_types': df.dtypes.to_dict(),
        }

    def _detect_csv_delimiter(self, file_path: str, encoding: str) -> str:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
        except Exception:
            return ','
        counts = {d: first_line.count(d) for d in (',', ';', '\t', '|')}
        valid = {d: c for d, c in counts.items() if c > 0}
        return max(valid, key=valid.get) if valid else ','

    def load_csv_file(self, file_path: str, table_name: str = None) -> bool:
        try:
            if not os.path.exists(file_path):
                print(f"CSV file not found: {file_path}")
                return False
            self.ensure_connection()

            df = None
            for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=enc,
                                     delimiter=self._detect_csv_delimiter(file_path, enc))
                    break
                except UnicodeDecodeError:
                    continue

            if df is None or df.empty:
                print(f"Could not read CSV or file is empty: {os.path.basename(file_path)}")
                return False

            df.columns = self._clean_column_names(df.columns)
            table_name = self._clean_table_name(
                table_name if table_name is not None
                else os.path.splitext(os.path.basename(file_path))[0]
            )
            self.register_table(
                table_name, df,
                source_file=os.path.basename(file_path), source_type='CSV',
            )
            self.loaded_files.append({
                'file_path': file_path,
                'type': 'CSV',
                'tables': [f"{os.path.basename(file_path)} -> {table_name}"],
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
                ef = pd.ExcelFile(file_path)
                sheet_names = ef.sheet_names
                ef.close()
            self.ensure_connection()

            loaded = []
            for sheet in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    if df.empty:
                        continue
                    df.columns = self._clean_column_names(df.columns)
                    tname = self._clean_table_name(sheet)
                    self.register_table(
                        tname, df,
                        source_file=os.path.basename(file_path), source_sheet=sheet,
                    )
                    loaded.append(f"{sheet} -> {tname}")
                except Exception as e:
                    print(f"Failed to load sheet '{sheet}': {e}")

            if loaded:
                self.loaded_files.append({
                    'file_path': file_path,
                    'type': 'Excel',
                    'sheets': loaded,
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
        if not col:  # blank/empty header cell
            col = 'unnamed_column'
        if col[0].isdigit():
            col = 'col_' + col
        return col.lower()

    def _clean_table_name(self, name: str) -> str:
        name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        if not name:  # blank/empty sheet or table name
            name = 'table'
        if name[0].isdigit():
            name = 'table_' + name
        return name.lower()

    @staticmethod
    def _dedupe_names(names: List[str]) -> List[str]:
        """Make a list of identifiers unique by suffixing collisions (_2, _3, ...).

        Two distinct source headers can clean to the same name (e.g. "Q1 Sales"
        and "Q1-Sales" both become "q1_sales"); without this, to_sql raises a
        duplicate-column error and the whole sheet fails to load.
        """
        seen: Dict[str, int] = {}
        out: List[str] = []
        for name in names:
            if name not in seen:
                seen[name] = 1
                out.append(name)
                continue
            seen[name] += 1
            candidate = f"{name}_{seen[name]}"
            while candidate in seen:
                seen[name] += 1
                candidate = f"{name}_{seen[name]}"
            seen[candidate] = 1
            out.append(candidate)
        return out

    def _clean_column_names(self, cols) -> List[str]:
        """Clean and de-duplicate a set of column names in one pass."""
        return self._dedupe_names([self._clean_column_name(c) for c in cols])

    def execute_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.connection)

    def get_schema_info(self) -> Dict:
        return {
            tname: [
                {'name': col,
                 'type': str(info['column_types'].get(col, 'TEXT')),
                 'nullable': True}
                for col in info['columns']
            ]
            for tname, info in self.tables_info.items()
        }

    def get_tables_summary(self) -> pd.DataFrame:
        rows = []
        for tname, info in self.tables_info.items():
            source = (
                f"{info['source_file']} (CSV)"
                if info.get('source_type') == 'CSV'
                else f"{info['source_file']} - Sheet: {info.get('source_sheet', '?')}"
            )
            rows.append({
                'Table': tname,
                'Source': source,
                'Rows': info['row_count'],
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
        self.azure_client = azure_client
        self.files_db = files_db
        self.deployment_name = deployment_name
        self.schema_info = self.files_db.get_schema_info()
        self.conversation_history: List[Dict] = []
        self.last_query_result = None
        self.last_query_context = None
        # Cache for the (expensive) sampled-schema block used by the
        # clarification step; keyed on a signature of the loaded tables.
        self._schema_samples_cache: Optional[str] = None
        self._schema_samples_sig: Optional[tuple] = None

    def refresh_schema_info(self) -> None:
        """Refresh cached schema metadata after the app mutates the in-memory DB."""
        self.schema_info = self.files_db.get_schema_info()
        # Loaded tables may have changed — drop the sampled-schema cache.
        self._schema_samples_cache = None
        self._schema_samples_sig = None

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def add_to_history(self, question: str, result: Dict):
        self.conversation_history.append({
            'question': question,
            'sql_query': result.get('sql_query'),
            'result_summary': (
                f"Returned {len(result.get('results', []))} rows"
                if result.get('success') else "Query failed"
            ),
        })
        self.last_query_result = result.get('results')
        self.last_query_context = {
            'previous_question': question,
            'previous_sql': result.get('sql_query'),
            'row_count': len(result.get('results', [])) if result.get('success') else 0,
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

    # Backslash is used as the LIKE ESCAPE character everywhere this is applied.
    _LIKE_ESCAPE = "\\"

    @classmethod
    def _escape_like_literal(cls, value: Any) -> str:
        """Escape a value for use inside a LIKE '%...%' pattern.

        Escapes the LIKE metacharacters % and _ (and the escape char itself) so a
        filter value like "50%" or "fy_2024" matches literally instead of acting
        as a wildcard, then escapes single quotes for the surrounding SQL literal.
        Callers must append  ESCAPE '\\'  to the LIKE clause.
        """
        text = str(value)
        text = text.replace(cls._LIKE_ESCAPE, cls._LIKE_ESCAPE * 2)
        text = text.replace("%", cls._LIKE_ESCAPE + "%").replace("_", cls._LIKE_ESCAPE + "_")
        return text.replace("'", "''")

    def _table_exists(self, table_name: str) -> bool:
        return table_name in self.files_db.tables_info

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        info = self.files_db.tables_info.get(table_name, {})
        return column_name in info.get('columns', [])

    def _distinct_nonempty_values(
            self,
            table_name: str,
            column_name: str,
            limit: int = 8,
    ) -> List[str]:
        """Small local value sampler used by deterministic guardrails."""
        if not self._table_exists(table_name) or not self._column_exists(table_name, column_name):
            return []
        try:
            q_table = self._quote_ident(table_name)
            q_col = self._quote_ident(column_name)
            sample_df = self.files_db.execute_query(
                f"SELECT DISTINCT TRIM(CAST({q_col} AS TEXT)) AS v "
                f"FROM {q_table} "
                f"WHERE {q_col} IS NOT NULL AND TRIM(CAST({q_col} AS TEXT)) <> '' "
                f"LIMIT {max(1, min(int(limit), 25))}"
            )
            values = []
            for value in sample_df.get("v", pd.Series(dtype="object")).tolist():
                text = str(value).strip()
                if text and text.lower() not in {"nan", "none", "null"}:
                    values.append(text)
            return values
        except Exception:
            return []

    def _financial_guardrail_warnings(
            self,
            table_name: str,
            value_column: str = "",
            group_columns: Optional[List[str]] = None,
            filter_columns: Optional[List[str]] = None,
            sql: str = "",
    ) -> List[str]:
        """Warn when financial-looking aggregations cross unsafe scopes."""
        info = self.files_db.tables_info.get(table_name, {})
        columns = [str(col) for col in info.get("columns", [])]
        if not columns:
            return []

        lower_columns = {col.lower(): col for col in columns}
        marker_names = {
            "value_numeric", "unit", "currency", "value_kind",
            "line_item", "line_item_path", "section", "period",
        }
        looks_financial = (
                bool(marker_names & set(lower_columns.keys()))
                or any("period" in col.lower() for col in columns)
                or any("currency" in col.lower() for col in columns)
                or str(value_column).lower() in {"value_numeric", "amount", "value"}
        )
        if not looks_financial:
            return []

        group_set = {str(col).lower() for col in (group_columns or []) if col}
        filter_set = {str(col).lower() for col in (filter_columns or []) if col}
        sql_lower = str(sql or "").lower()

        def scoped(column_name: str) -> bool:
            lower = column_name.lower()
            if lower in group_set or lower in filter_set:
                return True
            # Match the column as a whole token, not a substring, so that a
            # column named "unit" isn't considered scoped merely because
            # "business_unit" appears in the SQL (which would hide the warning).
            return bool(re.search(r"\b" + re.escape(lower) + r"\b", sql_lower))

        warnings_out: List[str] = []
        for candidate in ("unit", "currency", "value_kind"):
            col = lower_columns.get(candidate)
            if not col or scoped(col):
                continue
            values = self._distinct_nonempty_values(table_name, col, limit=6)
            if len(values) > 1:
                warnings_out.append(
                    f'Table "{table_name}" has multiple {col} values ({", ".join(values[:5])}); '
                    f"filter or group by {col} before trusting an aggregate."
                )

        period_cols = [
            col for col in columns
            if col.lower() in {"period", "year", "date", "fiscal_year", "fiscal_period"}
               or "period" in col.lower()
        ]
        for col in period_cols[:2]:
            if scoped(col):
                continue
            values = self._distinct_nonempty_values(table_name, col, limit=6)
            if len(values) > 1:
                warnings_out.append(
                    f'Table "{table_name}" spans multiple {col} values ({", ".join(values[:5])}); '
                    f"aggregate by or filter {col} when comparing periods."
                )
                break

        isolated_col = lower_columns.get("is_isolated_cell")
        if isolated_col and not scoped(isolated_col):
            isolated_values = self._distinct_nonempty_values(table_name, isolated_col, limit=3)
            if any(str(value).strip() == "1" for value in isolated_values):
                warnings_out.append(
                    f'Table "{table_name}" contains isolated helper cells '
                    f"(is_isolated_cell = 1) — stray values far outside the main table; "
                    f"filter is_isolated_cell = 0 before trusting an aggregate."
                )

        hierarchy_cols = [
            lower_columns[name]
            for name in ("line_item", "line_item_path", "metric", "section", "label")
            if name in lower_columns
        ]
        subtotal_terms = ("total", "subtotal", "sub-total", "grand total")
        for col in hierarchy_cols[:3]:
            # A filtered or grouped hierarchy column cannot silently mix
            # detail rows with pre-aggregated totals: filtering pins the rows
            # to one label and grouping reports each label separately. Only
            # unscoped aggregates across the hierarchy are dangerous.
            if scoped(col):
                continue
            values = self._distinct_nonempty_values(table_name, col, limit=20)
            if any(any(term in value.lower() for term in subtotal_terms) for value in values):
                warnings_out.append(
                    f'Table "{table_name}" appears to contain subtotal/total rows in {col}; '
                    "avoid summing detail rows together with pre-aggregated totals."
                )
                break

        return list(dict.fromkeys(warnings_out))

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
        return _guided_result(
            question, 'guided_overview', success=True,
            sql_query=pseudo_sql, results=results, answer_summary=answer_summary,
            trace=[
                'Guided route: dataset overview',
                'Computed totals from files_db.tables_info',
            ],
            relevant_tables=list(tables_info.keys()),
            query_plan={'route': 'guided_overview'},
        )

    def _build_guided_tab_breakdown(self, question: str, sample_values_per_column: int = 3) -> Dict:
        """Return per-table column summaries with dtype and sample values."""
        tables_info = self.files_db.tables_info or {}
        rows: List[Dict[str, Any]] = []

        sample_limit = max(1, min(int(sample_values_per_column), 5))
        for table_name, info in tables_info.items():
            columns = list(info.get('columns', []))
            col_types = info.get('column_types', {})

            for col in columns:
                values = self._distinct_nonempty_values(table_name, col, limit=sample_limit)
                rows.append({
                    'table_name': table_name,
                    'row_count': int(info.get('row_count', 0)),
                    'column_name': str(col),
                    'column_type': str(col_types.get(col, 'TEXT')),
                    'sample_values': ", ".join(values[:sample_limit]),
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
        return _guided_result(
            question, 'guided_tab_breakdown', success=True,
            sql_query=pseudo_sql, results=results, answer_summary=answer_summary,
            trace=[
                'Guided route: tab breakdown',
                'Computed per-table column summaries with sample values',
            ],
            relevant_tables=list(tables_info.keys()),
            query_plan={
                'route': 'guided_tab_breakdown',
                'sample_values_per_column': sample_limit,
            },
        )

    def _execute_financial_guided_query(self, guided_request: Dict[str, Any]) -> Dict:
        """Execute deterministic financial workbook templates."""
        intent = str(guided_request.get('intent', '')).strip().lower()
        question = str(guided_request.get('question') or 'Guided financial query').strip()
        table_name = str(guided_request.get('table_name') or '').strip()

        def failure(message: str, trace_message: str) -> Dict[str, Any]:
            return _guided_result(
                question, f'guided_{intent}', success=False,
                error=message, trace=[trace_message],
                relevant_tables=[table_name] if table_name else [],
            )

        if not table_name or not self._table_exists(table_name):
            return failure('Invalid or missing table for guided financial query.',
                           'Guided financial route failed: invalid table')

        value_column = str(guided_request.get('value_column') or 'value_numeric').strip()
        item_column = str(guided_request.get('item_column') or '').strip()
        item_value = str(guided_request.get('item_value') or '').strip()
        section_column = str(guided_request.get('section_column') or '').strip()
        period_column = str(guided_request.get('period_column') or '').strip()
        period_a = str(guided_request.get('period_a') or '').strip()
        period_b = str(guided_request.get('period_b') or '').strip()
        limit = max(1, min(int(guided_request.get('limit') or 20), 500))
        allow_unsafe_aggregate = bool(guided_request.get('allow_unsafe_aggregate'))

        required_columns = [value_column]
        if intent in {'financial_line_item_over_time', 'financial_top_n', 'financial_compare_periods'}:
            required_columns.append(item_column)
        if intent in {'financial_line_item_over_time', 'financial_compare_periods'}:
            required_columns.append(period_column)
        if intent == 'financial_section_totals':
            required_columns.append(section_column)

        for column in required_columns:
            if not column or not self._column_exists(table_name, column):
                return failure(
                    f'Invalid or missing column "{column}" for guided financial query.',
                    'Guided financial route failed: invalid column',
                )

        if intent in {'financial_line_item_over_time', 'financial_compare_periods'} and not item_value:
            return failure(
                'Line item filter is required for this guided financial template.',
                'Guided financial route failed: missing line-item scope',
            )

        if intent == 'financial_compare_periods' and period_a and period_b and period_a == period_b:
            return failure(
                'Choose two different periods for comparison.',
                'Guided financial route failed: duplicate comparison periods',
            )

        q_table = self._quote_ident(table_name)
        q_value = self._quote_ident(value_column)
        where_parts: List[str] = []
        filter_columns: List[str] = []

        if item_value and item_column and intent in {'financial_line_item_over_time', 'financial_compare_periods'}:
            q_item = self._quote_ident(item_column)
            safe_item = self._escape_like_literal(item_value)
            where_parts.append(f"CAST({q_item} AS TEXT) LIKE '%{safe_item}%' ESCAPE '{self._LIKE_ESCAPE}'")
            filter_columns.append(item_column)

        for key in ('unit', 'currency', 'value_kind', 'group'):
            column = str(guided_request.get(f'{key}_column') or '').strip()
            value = str(guided_request.get(f'{key}_value') or '').strip()
            if column and value:
                if not self._column_exists(table_name, column):
                    return failure(
                        f'Invalid {key} filter column for guided financial query.',
                        'Guided financial route failed: invalid scope filter',
                    )
                q_col = self._quote_ident(column)
                safe_value = self._escape_sql_literal(value)
                where_parts.append(f"TRIM(CAST({q_col} AS TEXT)) = TRIM('{safe_value}')")
                filter_columns.append(column)

        period_filter_column = str(guided_request.get('period_filter_column') or '').strip()
        period_filter_value = str(guided_request.get('period_filter_value') or '').strip()
        if period_filter_column and period_filter_value:
            if not self._column_exists(table_name, period_filter_column):
                return failure(
                    'Invalid period filter column for guided financial query.',
                    'Guided financial route failed: invalid period filter',
                )
            q_period_filter = self._quote_ident(period_filter_column)
            safe_period_filter = self._escape_sql_literal(period_filter_value)
            where_parts.append(
                f"TRIM(CAST({q_period_filter} AS TEXT)) = TRIM('{safe_period_filter}')"
            )
            filter_columns.append(period_filter_column)

        where_clause = " WHERE " + " AND ".join(where_parts) if where_parts else ""
        group_columns: List[str] = []
        metric_alias = self._quote_ident('metric_value')

        if intent == 'financial_line_item_over_time':
            q_period = self._quote_ident(period_column)
            group_columns = [period_column]
            sql_query = (
                f"SELECT {q_period} AS {self._quote_ident(period_column)}, "
                f"SUM({q_value}) AS {metric_alias} "
                f"FROM {q_table}{where_clause} "
                f"GROUP BY {q_period} "
                f"ORDER BY {q_period} "
                f"LIMIT {limit}"
            )
            answer_summary = f"Line item trend returned up to {limit} period row(s) from '{table_name}'."
        elif intent == 'financial_section_totals':
            q_section = self._quote_ident(section_column)
            group_columns = [section_column]
            sql_query = (
                f"SELECT {q_section} AS {self._quote_ident(section_column)}, "
                f"SUM({q_value}) AS {metric_alias} "
                f"FROM {q_table}{where_clause} "
                f"GROUP BY {q_section} "
                f"ORDER BY {metric_alias} DESC "
                f"LIMIT {limit}"
            )
            answer_summary = f"Section totals returned up to {limit} row(s) from '{table_name}'."
        elif intent == 'financial_top_n':
            q_item = self._quote_ident(item_column)
            group_columns = [item_column]
            sql_query = (
                f"SELECT {q_item} AS {self._quote_ident(item_column)}, "
                f"SUM({q_value}) AS {metric_alias} "
                f"FROM {q_table}{where_clause} "
                f"GROUP BY {q_item} "
                f"ORDER BY {metric_alias} DESC "
                f"LIMIT {limit}"
            )
            answer_summary = f"Top {limit} financial rows returned from '{table_name}'."
        elif intent == 'financial_compare_periods':
            if not period_a or not period_b:
                return failure('Choose both comparison periods.', 'Guided financial route failed: missing periods')
            q_period = self._quote_ident(period_column)
            safe_a = self._escape_sql_literal(period_a)
            safe_b = self._escape_sql_literal(period_b)
            period_filter = f"TRIM(CAST({q_period} AS TEXT)) IN (TRIM('{safe_a}'), TRIM('{safe_b}'))"
            where_with_period = where_parts + [period_filter]
            compare_where = " WHERE " + " AND ".join(where_with_period)
            sql_query = (
                f"SELECT "
                f"SUM(CASE WHEN TRIM(CAST({q_period} AS TEXT)) = TRIM('{safe_a}') THEN {q_value} ELSE 0 END) AS {self._quote_ident(period_a)}, "
                f"SUM(CASE WHEN TRIM(CAST({q_period} AS TEXT)) = TRIM('{safe_b}') THEN {q_value} ELSE 0 END) AS {self._quote_ident(period_b)}, "
                f"SUM(CASE WHEN TRIM(CAST({q_period} AS TEXT)) = TRIM('{safe_b}') THEN {q_value} ELSE 0 END) - "
                f"SUM(CASE WHEN TRIM(CAST({q_period} AS TEXT)) = TRIM('{safe_a}') THEN {q_value} ELSE 0 END) AS {self._quote_ident('change')}"
                f" FROM {q_table}{compare_where}"
            )
            filter_columns.append(period_column)
            answer_summary = f"Compared {period_a} versus {period_b} from '{table_name}'."
        else:
            return failure('Unsupported guided financial intent.',
                           f"Guided financial route failed: unsupported intent '{intent}'")

        warnings_out = self._financial_guardrail_warnings(
            table_name,
            value_column=value_column,
            group_columns=group_columns,
            filter_columns=filter_columns,
            sql=sql_query,
        )
        query_plan = {
            'route': f'guided_{intent}',
            'table': table_name,
            'value_column': value_column,
            'item_column': item_column,
            'section_column': section_column,
            'period_column': period_column,
            'limit': limit,
        }

        if warnings_out and not allow_unsafe_aggregate:
            return _guided_result(
                question, f'guided_{intent}', success=False,
                sql_query=sql_query,
                error=(
                    'Financial guardrails blocked this aggregate. Add unit/currency/'
                    'period filters, narrow the line item, or explicitly choose to run '
                    'despite guardrail warnings.'
                ),
                warnings=warnings_out,
                trace=[
                    f"Guided route blocked: guided_{intent}",
                    f"Financial guardrails raised {len(warnings_out)} warning(s)",
                ],
                relevant_tables=[table_name],
                query_plan={**query_plan, 'blocked_by_guardrails': True},
            )

        try:
            results = self.files_db.execute_query(sql_query)
        except Exception as e:
            return _guided_result(
                question, f'guided_{intent}', success=False,
                sql_query=sql_query, error=str(e),
                trace=[f"Guided financial route execution failed: {e}"],
                relevant_tables=[table_name],
            )

        self.add_to_history(question, {
            'sql_query': sql_query,
            'results': results,
            'success': True,
        })
        return _guided_result(
            question, f'guided_{intent}', success=True,
            sql_query=sql_query, results=results, answer_summary=answer_summary,
            warnings=warnings_out,
            trace=[f"Guided route executed: guided_{intent}"],
            relevant_tables=[table_name],
            query_plan=query_plan,
        )

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
        if intent.startswith('financial_'):
            return self._execute_financial_guided_query(guided_request)

        table_name = str(guided_request.get('table_name') or '').strip()

        def failure(message: str, trace_message: str,
                    relevant_tables: Optional[List[str]] = None) -> Dict:
            return _guided_result(
                question, f'guided_{intent}', success=False,
                error=message, trace=[trace_message],
                relevant_tables=[table_name] if relevant_tables is None else relevant_tables,
            )

        if not table_name or not self._table_exists(table_name):
            # The table is unusable, so it is not reported as a relevant table.
            return failure('Invalid or missing table for guided query.',
                           'Guided route failed: invalid table', relevant_tables=[])

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
                return failure('Invalid or missing value column for guided query.',
                               'Guided route failed: invalid value column')
            agg_expr = f"{agg.upper()}({self._quote_ident(value_column)})"

        # Optional where clause
        where_clause = ''
        if where_column and where_value:
            if not self._column_exists(table_name, where_column):
                return failure('Invalid filter column for guided query.',
                               'Guided route failed: invalid filter column')
            q_filter_col = self._quote_ident(where_column)
            if where_mode == 'equals':
                safe_value = self._escape_sql_literal(where_value)
                where_clause = f" WHERE TRIM(CAST({q_filter_col} AS TEXT)) = TRIM('{safe_value}')"
            else:
                safe_value = self._escape_like_literal(where_value)
                where_clause = (
                    f" WHERE CAST({q_filter_col} AS TEXT) LIKE '%{safe_value}%' "
                    f"ESCAPE '{self._LIKE_ESCAPE}'"
                )

        q_table = self._quote_ident(table_name)
        metric_alias = self._quote_ident('metric_value')

        if intent == 'aggregate':
            sql_query = f"SELECT {agg_expr} AS {metric_alias} FROM {q_table}{where_clause}"
        elif intent == 'group_by':
            if not group_by_column or not self._column_exists(table_name, group_by_column):
                return failure('Invalid or missing group-by column for guided query.',
                               'Guided route failed: invalid group-by column')
            q_group = self._quote_ident(group_by_column)
            sql_query = (
                f"SELECT {q_group} AS {self._quote_ident(group_by_column)}, {agg_expr} AS {metric_alias} "
                f"FROM {q_table}{where_clause} "
                f"GROUP BY {q_group} "
                f"ORDER BY {metric_alias} DESC "
                f"LIMIT {limit}"
            )
        else:
            return failure('Unsupported guided intent.',
                           f"Guided route failed: unsupported intent '{intent}'")

        try:
            results = self.files_db.execute_query(sql_query)
            guardrail_warnings = self._financial_guardrail_warnings(
                table_name,
                value_column=value_column,
                group_columns=[group_by_column] if group_by_column else [],
                filter_columns=[where_column] if where_column else [],
                sql=sql_query,
            )
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
            return _guided_result(
                question, f'guided_{intent}', success=True,
                sql_query=sql_query, results=results, answer_summary=answer_summary,
                warnings=guardrail_warnings,
                trace=[f"Guided route executed: guided_{intent}"],
                relevant_tables=[table_name],
                query_plan={
                    'route': f'guided_{intent}',
                    'table': table_name,
                    'aggregation': agg,
                    'value_column': value_column,
                    'group_by_column': group_by_column,
                    'where_column': where_column,
                    'where_mode': where_mode,
                    'limit': limit,
                },
            )
        except Exception as e:
            return _guided_result(
                question, f'guided_{intent}', success=False,
                sql_query=sql_query, error=str(e),
                trace=[f"Guided route execution failed: {e}"],
                relevant_tables=[table_name],
            )

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str,
                  max_tokens: int = 800, temperature: float = 0.0) -> str:
        response = self.azure_client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _strip_fences(text: str, lang: str = "") -> str:
        text = re.sub(r"```" + lang + r"\s*", "", text, flags=re.IGNORECASE)
        return text.replace("```", "").strip()

    def _call_llm_json(self, system: str, user: str, max_tokens: int,
                       fallback: Optional[Dict] = None) -> Optional[Dict]:
        """LLM call expected to return a JSON object; returns `fallback` on any failure."""
        try:
            parsed = json.loads(
                self._strip_fences(self._call_llm(system, user, max_tokens=max_tokens), "json")
            )
        except Exception:
            return fallback
        return parsed if isinstance(parsed, dict) else fallback

    def _schema_text(self, schema: Dict, schema_context: Optional[str],
                     include_samples: bool) -> str:
        """Routed schema contract when the orchestrator supplied one, else the
        locally rendered schema block."""
        return schema_context or self._format_schema_for_prompt(
            schema, include_samples=include_samples
        )

    @staticmethod
    def _plan_json(plan: QueryPlan) -> str:
        """The plan as JSON for prompts — everything except free-text notes."""
        return json.dumps(
            {k: v for k, v in asdict(plan).items() if k != "notes"}, indent=2
        )

    @staticmethod
    def _intent_from_dict(d: Dict) -> ParsedIntent:
        filters = d.get("filters")
        return ParsedIntent(
            action=d.get("action", "list"),
            entities=[str(e).lower() for e in (d.get("entities") or [])],
            filters=filters if isinstance(filters, dict) else {},
            aggregation=d.get("aggregation", "none"),
            group_by_hint=d.get("group_by_hint"),
            sort_hint=d.get("sort_hint"),
            sort_order=d.get("sort_order", "asc"),
            limit=d.get("limit"),
            raw=d,
        )

    @staticmethod
    def _plan_from_dict(d: Dict) -> QueryPlan:
        def as_list(key, default):
            value = d.get(key, default)
            return value if isinstance(value, list) else default

        return QueryPlan(
            tables=as_list("tables", []),
            columns=as_list("columns", ["*"]),
            filters=as_list("filters", []),
            aggregation=d.get("aggregation"),
            group_by=as_list("group_by", []),
            order_by=d.get("order_by"),
            limit=d.get("limit"),
            joins=as_list("joins", []),
            notes=d.get("notes", ""),
        )

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

        The result is cached per loaded-table signature: this block issues one or
        two SQLite queries per text column across *every* table, and the
        clarification step runs on every free-text question, so recomputing it
        each time is the main avoidable cost in the pipeline.
        """
        signature = tuple(
            (tname, int(info.get('row_count', 0)), tuple(info.get('columns', [])))
            for tname, info in self.files_db.tables_info.items()
        )
        if (
                self._schema_samples_cache is not None
                and self._schema_samples_sig == signature
        ):
            return self._schema_samples_cache

        lines = []
        for tname, info in self.files_db.tables_info.items():
            lines.append(f"Table: {tname} [{info['row_count']} rows]")
            col_types = info.get('column_types', {})

            for col in info['columns']:
                dtype = str(col_types.get(col, 'TEXT'))
                if not any(k in dtype.lower() for k in ('object', 'str', 'text')):
                    lines.append(f"  - {col} ({dtype})")
                    continue

                try:
                    q_tname = self._quote_ident(tname)
                    q_col = self._quote_ident(col)
                    n_distinct = int(self.files_db.execute_query(
                        f'SELECT COUNT(DISTINCT TRIM({q_col})) as n FROM {q_tname}'
                    )['n'].iloc[0])
                    low_cardinality = n_distinct <= 30
                    vdf = self.files_db.execute_query(
                        f'SELECT DISTINCT TRIM({q_col}) as v FROM {q_tname} '
                        f'WHERE {q_col} IS NOT NULL'
                        + (' ORDER BY v LIMIT 20' if low_cardinality else ' LIMIT 3')
                    )
                    vals = [str(v) for v in vdf['v'].tolist()
                            if str(v) not in ('nan', 'None', '')]
                except Exception:
                    lines.append(f"  - {col} ({dtype})")
                    continue

                if not low_cardinality:
                    lines.append(
                        f"  - {col} ({dtype}): {_quoted_list(vals)}, "
                        f"... ({n_distinct} distinct values)"
                    )
                    continue

                col_line = f"  - {col} ({dtype}): {_quoted_list(vals[:15])}"
                suspect = self._grouping_row_suspects(vals)
                if suspect:
                    col_line += (
                        f'  ⚠️ possible grouping/subtotal rows: '
                        f'{_quoted_list(suspect[:3])}'
                    )
                lines.append(col_line)

            lines.append("")

        rendered = "\n".join(lines)
        self._schema_samples_cache = rendered
        self._schema_samples_sig = signature
        return rendered

    def _grouping_row_suspects(self, vals: List[str]) -> List[str]:
        """Values that look like subtotal/grouping rows rather than detail categories."""
        # Check 1: value contains an aggregation-hint word.
        grouping_vals = [
            v for v in vals
            if any(hint in v.lower() for hint in self._GROUPING_ROW_HINTS)
        ]
        # Check 2: value is a prefix of another value
        # (e.g. "Fixed Income" and "Fixed Income Securities").
        prefix_vals = [
            v for v in vals
            if any(
                other.lower().startswith(v.lower() + " ")
                for other in vals
                if other != v and len(v) > 3
            )
        ]
        return list(dict.fromkeys(grouping_vals + prefix_vals))

    def summarize_tables(self, use_ai: bool = True) -> Dict[str, Dict[str, str]]:
        """One plain-language sentence per loaded table.

        Returns {table_name: {"summary": str, "source": "ai"|"heuristic"}}.
        A deterministic heuristic summary is always produced first, so the
        result is useful even with no LLM available; when use_ai is set and an
        Azure client is present, a single batched LLM call refines the wording
        for every table at once (content-aware, using the sampled schema).
        """
        tables_info = self.files_db.tables_info or {}
        overviews: Dict[str, Dict[str, str]] = {
            table_name: {
                "summary": heuristic_table_summary(table_name, info),
                "source": "heuristic",
            }
            for table_name, info in tables_info.items()
        }
        if not overviews or not use_ai or self.azure_client is None:
            return overviews

        try:
            schema_block = self._get_schema_with_samples()
        except Exception:
            return overviews

        system = (
            "You describe database tables for a business analyst. For EACH table, "
            "write ONE concise sentence (max ~22 words) stating what real-world "
            "information it holds: the business domain, the main breakdown "
            "dimensions (e.g. by segment, region, period), and the time basis if "
            "evident (year-to-date, quarter). Do not list columns or mention SQL. "
            'Return ONLY a JSON object mapping each exact table name to its sentence.'
        )
        user = (
            "Tables (with sample values):\n"
            f"{schema_block}\n\n"
            f"Table names: {', '.join(overviews.keys())}\n"
            "Return only the JSON object."
        )
        try:
            raw = self._strip_fences(self._call_llm(system, user, max_tokens=900), "json")
            parsed = json.loads(raw)
        except Exception:
            return overviews

        if isinstance(parsed, dict):
            for table_name in overviews:
                sentence = parsed.get(table_name)
                if isinstance(sentence, str) and sentence.strip():
                    overviews[table_name] = {
                        "summary": sentence.strip(),
                        "source": "ai",
                    }
        return overviews

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
        # table/column arrive from LLM output (which reads untrusted workbook
        # content), so they must be whitelisted against the loaded schema and
        # quoted — never trusted as raw SQL fragments.
        if not self._table_exists(table) or not self._column_exists(table, column):
            return []

        q_table = self._quote_ident(table)
        q_col = self._quote_ident(column)
        term = search_term.strip()
        safe = self._escape_sql_literal(term)
        safe_like = self._escape_like_literal(term)
        like_escape = f" ESCAPE '{self._LIKE_ESCAPE}'"
        found: Dict[str, int] = {}  # value → best tier

        def _run(where: str, limit: int = 20) -> List[str]:
            try:
                df = self.files_db.execute_query(
                    f"SELECT DISTINCT TRIM({q_col}) as v FROM {q_table} "
                    f"WHERE {where} AND {q_col} IS NOT NULL ORDER BY v LIMIT {limit}"
                )
                return [str(v) for v in df["v"].tolist()
                        if str(v) not in ("nan", "None", "")]
            except Exception:
                return []

        tier_clauses = [
            f"LOWER(TRIM({q_col})) = LOWER('{safe}')",                        # 0: exact
            f"LOWER(TRIM({q_col})) LIKE LOWER('{safe_like}%'){like_escape}",  # 1: starts with
            f"{q_col} LIKE '%{safe_like}%'{like_escape}",                     # 2: contains term
        ]
        # Tier 3: individual words (for multi-word terms like "Allianz Group")
        tier_clauses += [
            f"{q_col} LIKE '%{self._escape_like_literal(word)}%'{like_escape}"
            for word in re.split(r"\s+", term) if len(word) > 2
        ]

        for tier, where in enumerate(tier_clauses):
            for v in _run(where):
                found.setdefault(v, min(tier, 3))

        if not found:
            return []

        # Sort by tier, then alphabetically within the same tier
        ranked = sorted(found.items(), key=lambda x: (x[1], x[0]))
        return [(v, "") for v, _ in ranked[:8]]  # return up to 8; caller will trim to 5

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

        parsed = self._call_llm_json(system, user, max_tokens=600)
        if not parsed or not parsed.get("needs_clarification"):
            return None

        # ── Post-process options ───────────────────────────────────────────
        llm_options: List[str] = parsed.get("clarification_options") or []
        llm_labels: List[str] = parsed.get("option_labels") or []
        llm_labels = (llm_labels + [""] * len(llm_options))[:len(llm_options)]
        llm_pairs = list(zip(llm_options, llm_labels))

        ambiguity_type = parsed.get("ambiguity_type", "non_trivial_assumption")
        lookup = parsed.get("lookup") or {}
        table = lookup.get("table", "")
        column = lookup.get("column", "")
        term = parsed.get("ambiguous_term") or ""

        # DB enrichment applies to ALL types that provide a lookup, not just fuzzy_entity.
        # This is the key fix: non_trivial_assumption now gets real DB values too.
        merged = llm_pairs
        if table and column and term and table in self.files_db.tables_info:
            db_vals = [v for v, _ in self._fetch_ranked_entity_options(table, column, term)]

            if ambiguity_type == "fuzzy_entity":
                # Keep labeled conceptual options; unlabeled ones give way to DB values.
                merged = [(o, l) for o, l in llm_pairs if l]
            elif ambiguity_type in ("non_trivial_assumption", "missing_filter_value"):
                # LLM options that exactly match a real DB value are schema-grounded and
                # kept; purely invented ones give way to the ranked DB values.
                merged = [(o, l) for o, l in llm_pairs if o in set(db_vals)]
            elif ambiguity_type == "mixed":
                # Keep LLM structured options, appending DB values if there is room.
                merged = llm_pairs
            else:
                db_vals = []

            kept = {o for o, _ in merged}
            merged = merged + [(v, "") for v in db_vals if v not in kept]

        final_opts = [o for o, _ in merged[:5]]
        final_labels = [l for _, l in merged[:5]]

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
        return self._intent_from_dict(self._call_llm_json(
            system, f"Parse this data query:\n{question}", max_tokens=350,
            fallback={
                "action": "list", "entities": [], "filters": {},
                "aggregation": "none", "group_by_hint": None,
                "sort_hint": None, "sort_order": "asc", "limit": None,
            },
        ))

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
            score = 0
            ttokens = set(re.split(r'[_\s]', tname.lower()))
            score += len(tokens & ttokens) * 3
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
        relevant = {t: self.schema_info[t] for t, s in scores.items() if s >= threshold}
        if not relevant:
            best = max(scores, key=scores.get)
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
            info = self.files_db.tables_info.get(tname, {})
            col_lines = "\n".join(f"  - {c['name']} ({c['type']})" for c in columns)
            block = (
                f"Table: {tname}  [source: {info.get('source_file', '?')}, "
                f"rows: {info.get('row_count', '?')}]\n{col_lines}"
            )
            if include_samples:
                try:
                    q_table = self._quote_ident(tname)
                    sample_df = self.files_db.execute_query(
                        f"SELECT * FROM {q_table} LIMIT 3"
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
            self, question: str, intent: ParsedIntent, schema: Dict,
            schema_context: Optional[str] = None
    ) -> QueryPlan:
        schema_text = self._schema_text(schema, schema_context, include_samples=True)
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
        return self._plan_from_dict(self._call_llm_json(
            system, user, max_tokens=600,
            fallback={
                "tables": list(schema.keys())[:1], "columns": ["*"],
                "filters": [], "aggregation": None, "group_by": [],
                "order_by": None, "limit": 100, "joins": [],
                "notes": "fallback plan",
            },
        ))

    # ------------------------------------------------------------------
    # Step 4 — SQL generation from plan
    # ------------------------------------------------------------------

    def generate_sql_from_plan(
            self, question: str, intent: ParsedIntent,
            plan: QueryPlan, schema: Dict,
            schema_context: Optional[str] = None
    ) -> str:
        schema_text = self._schema_text(schema, schema_context, include_samples=False)
        plan_json = self._plan_json(plan)
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
    # Steps 1–4 fused — intent + plan + SQL in a single LLM round-trip
    # ------------------------------------------------------------------

    def generate_plan_and_sql(
            self,
            question: str,
            schema: Dict,
            schema_context: Optional[str] = None,
    ) -> Tuple[ParsedIntent, QueryPlan, str]:
        """Produce intent, query plan, and SQL in one call.

        The legacy pipeline spent three sequential LLM round-trips on
        parse_intent → build_query_plan → generate_sql_from_plan. Because the
        SQL is derived from the plan and the plan from the intent, a single
        structured completion can emit all three coherently — the plan acts as
        in-completion scaffolding for the SQL, so they cannot drift apart, and
        the two extra round-trips (and their latency) disappear. Downstream
        deterministic checks still receive real ParsedIntent / QueryPlan
        objects. run_query falls back to the granular methods if this fails.
        """
        schema_text = self._schema_text(schema, schema_context, include_samples=True)
        system = (
            "You are a SQLite analytics planner. In ONE response you interpret "
            "the question, produce a query plan, and write the final SQLite SQL.\n"
            "Return ONLY a valid JSON object with exactly these keys:\n"
            "{\n"
            '  "intent": {\n'
            '    "action": one of ["aggregate","filter","list","count","compare","lookup"],\n'
            '    "entities": [key nouns/concepts, lowercase],\n'
            '    "aggregation": one of ["sum","count","avg","max","min","none"],\n'
            '    "group_by_hint": string or null,\n'
            '    "sort_hint": string or null,\n'
            '    "sort_order": "asc" or "desc",\n'
            '    "limit": integer or null\n'
            "  },\n"
            '  "plan": {\n'
            '    "tables": [table names to use],\n'
            '    "columns": [columns to SELECT — use * only as last resort],\n'
            '    "filters": [{"column":"...","operator":"=|!=|>|<|>=|<=|LIKE|IS NULL","value":"..."}],\n'
            '    "aggregation": {"function":"SUM|COUNT|AVG|MAX|MIN","column":"..."} or null,\n'
            '    "group_by": [column names],\n'
            '    "order_by": {"column":"...","direction":"ASC|DESC"} or null,\n'
            '    "limit": integer or null,\n'
            '    "joins": [{"left_table":"...","right_table":"...","left_col":"...","right_col":"..."}],\n'
            '    "notes": "brief notes"\n'
            "  },\n"
            '  "sql": "one valid SQLite SELECT or WITH query, consistent with plan"\n'
            "}\n"
            "SQL rules:\n"
            "- Use exact table and column names from the schema\n"
            "- SQLite syntax only (no ILIKE, no ::cast)\n"
            "- Use TRIM() for text columns that may have extra spaces\n"
            "- Use LIKE '%value%' for partial matches\n"
            "- Only SELECT or WITH queries are allowed\n"
            "- sql MUST match plan.tables / filters / aggregation / group_by\n"
            "No explanation. No markdown. Only the JSON object."
        )
        user = (
            f"Schema:\n{schema_text}\n\n"
            f"{self.get_conversation_context()}\n"
            f"User question: {question}\n\n"
            "Return the intent, plan, and SQL as one JSON object:"
        )
        raw = self._call_llm(system, user, max_tokens=4000)
        raw = self._strip_fences(raw, "json")
        d = json.loads(raw)

        intent_d = d.get("intent") or {}
        plan_d = d.get("plan") or {}
        intent = self._intent_from_dict(intent_d if isinstance(intent_d, dict) else {})
        plan = self._plan_from_dict(plan_d if isinstance(plan_d, dict) else {})
        sql = self._strip_fences(str(d.get("sql", "") or ""), "sql").strip()
        return intent, plan, sql

    # ------------------------------------------------------------------
    # Step 5 — Deterministic SQL validation
    # ------------------------------------------------------------------

    # Quoted string literals / identifiers, so keyword and statement-separator
    # scans never match against data values or column names.
    _SQL_QUOTED_RE = re.compile(
        r"'(?:[^']|'')*'"  # single-quoted string literal
        r'|"(?:[^"]|"")*"'  # double-quoted identifier
        r"|`(?:[^`]|``)*`"  # backtick identifier
        r"|\[[^\]]*\]"  # bracketed identifier
    )

    @classmethod
    def _strip_sql_literals(cls, sql: str) -> str:
        """Blank the *interior* of quoted strings/identifiers (length-preserving).

        Keeps delimiters and character positions intact so the returned string
        can be scanned for keywords and split on statement-separating semicolons
        without matching anything that lives inside a data value like 'Call'.
        """

        def _blank(match: "re.Match[str]") -> str:
            token = match.group(0)
            return token[0] + ("x" * (len(token) - 2)) + token[-1]

        return cls._SQL_QUOTED_RE.sub(_blank, sql)

    def validate_sql(
            self, sql: str, plan: QueryPlan
    ) -> Tuple[bool, str, List[str]]:
        warn = []
        cleaned = sql.strip().rstrip(";").strip()
        # Scan against a copy with literals/identifiers blanked out so that a
        # value such as WHERE type = 'Call' is not mistaken for a CALL keyword.
        scan = self._strip_sql_literals(cleaned)
        upper = scan.upper().lstrip()

        if not (upper.startswith("SELECT") or upper.startswith("WITH")):
            return False, cleaned, ["SQL does not start with SELECT or WITH — blocked."]

        for kw in _DANGEROUS_KEYWORDS:
            # Allow SQLite string REPLACE() function in read-only SELECT queries.
            # Still block mutating forms such as REPLACE INTO.
            if kw == "REPLACE":
                if re.search(r"\bREPLACE\s+INTO\b", scan, re.IGNORECASE):
                    return False, cleaned, ["Dangerous keyword 'REPLACE INTO' found — blocked."]
                if re.search(r"\bREPLACE\s*\(", scan, re.IGNORECASE):
                    continue

            if re.search(r'\b' + kw + r'\b', scan, re.IGNORECASE):
                return False, cleaned, [f"Dangerous keyword '{kw}' found — blocked."]

        # SQLite functions that reach outside the in-memory database. They are
        # callable from a plain SELECT, so the statement-keyword scan above
        # never catches them.
        for fn in ("load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"):
            if re.search(r'\b' + fn + r'\s*\(', scan, re.IGNORECASE):
                return False, cleaned, [f"Dangerous function '{fn}()' found — blocked."]

        # A semicolon inside a literal is data, not a statement separator; only
        # cut at a real one (found via the blanked scan) to avoid truncating SQL.
        semicolon_idx = scan.find(";")
        if semicolon_idx != -1:
            cleaned = cleaned[:semicolon_idx].strip()
            scan = scan[:semicolon_idx]
            warn.append("Multiple SQL statements detected; only the first was kept.")

        has_limit = bool(re.search(r'\bLIMIT\b', scan, re.IGNORECASE))
        has_agg = plan.aggregation is not None or bool(plan.group_by)
        has_agg_fn = bool(re.search(
            r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', scan, re.IGNORECASE
        ))
        if not has_limit and not has_agg and not has_agg_fn:
            cleaned += " LIMIT 1000"
            warn.append("No LIMIT on non-aggregate query — LIMIT 1000 applied automatically.")

        return True, cleaned, warn

    def check_sql_semantics(
            self,
            sql: str,
            plan: QueryPlan,
            schema: Dict[str, List[Dict]],
    ) -> List[str]:
        """Deterministic SQL quality checks before execution."""
        warnings: List[str] = []
        sql_text = str(sql or "")
        upper = sql_text.upper()

        schema_tables = set(schema.keys())
        known_tables = set(self.files_db.tables_info.keys())
        planned_tables = [str(t) for t in (plan.tables or []) if str(t)]

        for table in planned_tables:
            if table not in known_tables:
                warnings.append(f'Planned table "{table}" is not loaded in the active database.')
            elif schema_tables and table not in schema_tables:
                warnings.append(f'Planned table "{table}" was not in the routed schema context.')

        if re.search(r"\bFROM\s+schema\b|\bJOIN\s+schema\b", sql_text, re.IGNORECASE):
            warnings.append("SQL appears to reference the Schema metadata table; this should normally be avoided.")

        if re.search(r"\bSELECT\s+\*", sql_text, re.IGNORECASE):
            for table in planned_tables:
                info = self.files_db.tables_info.get(table, {})
                col_count = len(info.get("columns", []))
                if col_count >= 20:
                    warnings.append(
                        f'SQL uses SELECT * on wide table "{table}" ({col_count} columns); '
                        "selecting specific columns is usually safer."
                    )

        has_join = bool(re.search(r"\bJOIN\b", upper))
        if has_join and not plan.joins:
            warnings.append("SQL contains a JOIN, but the query plan did not explicitly require one.")
        if len(planned_tables) > 1 and not (plan.joins or has_join):
            warnings.append("Multiple tables were planned but no join relationship is specified.")

        agg_match = re.search(
            r"\b(?:SUM|AVG|MIN|MAX)\s*\(\s*\"?([A-Za-z_][A-Za-z0-9_]*)\"?\s*\)",
            sql_text,
            re.IGNORECASE,
        )
        agg_col = agg_match.group(1) if agg_match else ""
        if agg_match:
            for table in planned_tables:
                info = self.files_db.tables_info.get(table, {})
                if agg_col in info.get("columns", []):
                    dtype = str(info.get("column_types", {}).get(agg_col, ""))
                    if not any(token in dtype.lower() for token in ("int", "float", "decimal", "number")):
                        warnings.append(
                            f'Aggregation uses "{agg_col}" with dtype "{dtype}". '
                            "If this is a formatted text amount, use a numeric value column instead."
                        )
                    break

        has_agg_fn = bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", upper))
        if has_agg_fn and re.search(r"\bGROUP\s+BY\b", upper) is None:
            selected_expr = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql_text, re.IGNORECASE | re.DOTALL)
            if selected_expr:
                select_text = selected_expr.group(1)
                comma_count = select_text.count(",")
                if comma_count > 0 and not re.search(r"\bOVER\s*\(", select_text, re.IGNORECASE):
                    warnings.append(
                        "SQL mixes aggregate and non-aggregate select expressions without GROUP BY."
                    )

        if has_agg_fn:
            for table in planned_tables:
                if table in known_tables:
                    warnings.extend(
                        self._financial_guardrail_warnings(
                            table,
                            value_column=agg_col,
                            group_columns=list(plan.group_by or []),
                            filter_columns=[
                                str(filter_item.get("column", ""))
                                for filter_item in (plan.filters or [])
                                if str(filter_item.get("column", ""))
                            ],
                            sql=sql_text,
                        )
                    )

        for filter_item in plan.filters or []:
            col = str(filter_item.get("column", ""))
            op = str(filter_item.get("operator", "")).strip()
            if op == "=" and any(hint in col.lower() for hint in self._ENTITY_COLUMN_HINTS):
                warnings.append(
                    f'Exact match filter on entity-like column "{col}" may miss close spellings; '
                    "consider whether LIKE/contains is intended."
                )

        return list(dict.fromkeys(warnings))

    # ------------------------------------------------------------------
    # Step 6 — SQL repair
    # ------------------------------------------------------------------

    def _repair_sql(
            self, question: str, intent: ParsedIntent, plan: QueryPlan,
            failed_sql: str, error_msg: str, schema: Dict,
            schema_context: Optional[str] = None
    ) -> str:
        schema_text = self._schema_text(schema, schema_context, include_samples=False)
        plan_json = self._plan_json(plan)
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
        equality_filters = [
            f for f in (plan.filters or [])
            if str(f.get("operator", "")).strip() == "=" and str(f.get("column", ""))
        ]
        exact_text_filters = [
            f for f in equality_filters
            if any(
                hint in str(f.get("column", "")).lower()
                for hint in self._ENTITY_COLUMN_HINTS
            )
        ]
        # An entity-looking column is the strongest signal, but a query that
        # returned nothing is worth diagnosing whatever the filter column is
        # called — that is exactly when a spelling suggestion helps most.
        if not exact_text_filters and flags.empty_result:
            exact_text_filters = equality_filters

        if exact_text_filters and (flags.empty_result or flags.suspicious_zero_result):
            # Pick the first entity-like filter to investigate.
            suspect = exact_text_filters[0]
            col = str(suspect.get("column", ""))
            val = str(suspect.get("value", ""))
            flags.filter_column = col
            flags.filter_value = val

            if col and val:
                flags.possible_exact_match_miss = True

                # Rank the real values by closeness to the one that missed.
                # This reuses the clarification lookup, which also matches on
                # individual words — so "Allianz Group" surfaces "Allianz SE"
                # even though no stored value contains that whole string.
                for tname in plan.tables:
                    if col not in [c['name'] for c in self.schema_info.get(tname, [])]:
                        continue
                    similar = [
                        value
                        for value, _ in self._fetch_ranked_entity_options(tname, col, val)
                        if value.strip().lower() != val.strip().lower()
                    ]
                    if similar:
                        flags.entity_match_uncertain = True
                        flags.similar_values = similar[:5]
                        break  # found results from this table; stop

        return flags

    # ------------------------------------------------------------------
    # Step 7b — Answer summary  (calibrated to ResultFlags)
    # ------------------------------------------------------------------

    # Columns that qualify what a figure means; quoting a number without them
    # is what makes a mixed-scope answer look authoritative but wrong.
    _SCOPE_REPORT_COLUMNS = ("unit", "currency", "value_kind", "period",
                             "section", "column_group", "segment")

    def result_scope_facts(self, results: pd.DataFrame,
                           plan: Optional[QueryPlan] = None) -> Dict[str, List[str]]:
        """Scope values carried by a result, e.g. {"unit": ["EUR mn"]}.

        Read from the result frame when it carries the columns; otherwise from
        the planned tables, so an aggregate that dropped its unit column can
        still be reported with the unit it was computed in.
        """
        facts: Dict[str, List[str]] = {}
        if results is None or results.empty:
            return facts

        lower_columns = {str(c).lower(): str(c) for c in results.columns}
        for name in self._SCOPE_REPORT_COLUMNS:
            column = lower_columns.get(name)
            if column is None:
                continue
            values = [
                str(v).strip() for v in results[column].dropna().unique()
                if str(v).strip() and str(v).strip().lower() not in ("nan", "none")
            ]
            if values:
                facts[name] = sorted(values)[:6]

        # Fall back to the source tables for scope the SELECT did not carry.
        for tname in (plan.tables if plan else []):
            if tname not in self.files_db.tables_info:
                continue
            table_columns = {
                str(c).lower(): str(c)
                for c in self.files_db.tables_info[tname].get("columns", [])
            }
            for name in ("unit", "currency", "value_kind"):
                if name in facts:
                    continue
                column = table_columns.get(name)
                if not column:
                    continue
                values = self._distinct_nonempty_values(tname, column, limit=6)
                # Only useful when the whole table shares one value; otherwise
                # the aggregate genuinely spans several and we must not claim one.
                if len(values) == 1:
                    facts[name] = values

        return facts

    @staticmethod
    def format_scope_facts(facts: Dict[str, List[str]]) -> str:
        """Render scope facts as 'unit: EUR mn · currency: EUR'."""
        return " · ".join(
            f"{name}: {', '.join(values)}"
            for name, values in facts.items() if values
        )

    def _generate_answer_summary(
            self, question: str, sql: str,
            results: pd.DataFrame, flags: ResultFlags,
            plan: Optional[QueryPlan] = None
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
                    alts = _quoted_list(flags.similar_values)
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
                    alts = _quoted_list(flags.similar_values)
                    base += f"Similar values in the data include: {alts}."
            else:
                base += (
                    "This may indicate a true zero value, or that the matched rows "
                    "contain null / zero entries for the requested field."
                )
            return base

        # ── Normal result: use LLM summary but with care instructions ──────
        sample = results.head(10).to_string(index=False)
        is_suspicious = flags.possible_exact_match_miss or flags.entity_match_uncertain

        scope_facts = self.result_scope_facts(results, plan)
        scope_text = self.format_scope_facts(scope_facts)
        scope_rule = (
            "- ALWAYS state the unit and currency next to any figure you quote "
            f"(the result's scope is: {scope_text}). A number without its unit "
            "is not an answer.\n"
            if scope_text else ""
        )

        if is_suspicious:
            system = (
                "You are a careful data analyst assistant. "
                "Write a concise (2–3 sentences) plain-language answer. "
                "IMPORTANT RULES:\n"
                f"{scope_rule}"
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
                "addresses the user's question using the query results.\n"
                f"{scope_rule}"
                "No markdown, no bullet points, no preamble."
            )

        user = (
            f"Question: {question}\n\nSQL:\n{sql}\n\n"
            + (f"Result scope: {scope_text}\n\n" if scope_text else "")
            + f"Result ({len(results)} rows total, showing up to 10):\n{sample}\n\nAnswer:"
        )
        try:
            summary = self._call_llm(system, user, max_tokens=250, temperature=0.2)
        except Exception:
            summary = f"The query returned {len(results)} row(s)."

        # Deterministic backstop: if the model quoted figures without naming the
        # unit, append it rather than leaving a bare number on screen. Only
        # unit-like axes count — a period such as "2023" appears in the prose
        # naturally and must not be mistaken for the figure's unit.
        measure_facts = {
            name: values for name, values in scope_facts.items()
            if name in ("unit", "currency", "value_kind")
        }
        if measure_facts and not any(
                value.lower() in summary.lower()
                for values in measure_facts.values() for value in values
        ):
            summary = f"{summary.rstrip()} (Scope — {scope_text}.)"
        return summary

    # ------------------------------------------------------------------
    # Steps 1–7: run_query
    # ------------------------------------------------------------------

    def run_query(
            self,
            question: str,
            schema_context: Optional[str] = None,
    ) -> QueryResponse:
        """Full pipeline — called only after clarification check returns None."""
        trace: List[str] = []
        warn: List[str] = []

        if schema_context:
            trace.append("Context: using routed schema contract")

        intent: Optional[ParsedIntent] = None
        plan: Optional[QueryPlan] = None
        sql: Optional[str] = None
        relevant_schema: Dict = {}
        repair_attempts = 0

        def failed(error: str, sql_query: Optional[str] = None) -> QueryResponse:
            return QueryResponse(
                question=question, interpreted_intent=intent.raw,
                relevant_tables=list(relevant_schema.keys()), query_plan=asdict(plan),
                sql_query=sql_query, results=None, answer_summary="",
                warnings=warn, trace=trace, success=False, error=error,
                repair_attempts=repair_attempts,
            )

        # ── Steps 1–4 (fast path): intent + plan + SQL in ONE LLM call ──────
        trace.append("Steps 1-4: Combined intent + plan + SQL generation")
        try:
            intent, plan, sql = self.generate_plan_and_sql(
                question, self.schema_info, schema_context=schema_context
            )
            if not sql:
                raise ValueError("combined generation returned no SQL")
            # Downstream checks/display want the schema for the planned tables.
            relevant_schema = {
                                  t: self.schema_info[t]
                                  for t in (plan.tables or [])
                                  if t in self.schema_info
                              } or dict(self.schema_info)
            trace.append(
                f"  → action={intent.action}, agg={intent.aggregation}, "
                f"tables={plan.tables}, {len(sql)} chars SQL"
            )
        except Exception as e:
            trace.append(f"  → combined path unavailable ({e}); using granular steps")
            intent, plan, sql = None, None, None

        # ── Granular fallback: original Steps 1–4, one LLM call each ────────
        if sql is None:
            # Steps 1-3 cannot raise: parse_intent and build_query_plan fall back
            # to a default dict internally, and select_relevant_schema is pure Python.
            trace.append("Step 1: Parsing intent")
            intent = self.parse_intent(question)
            trace.append(f"  → action={intent.action}, agg={intent.aggregation}, entities={intent.entities}")

            trace.append("Step 2: Selecting relevant schema")
            relevant_schema = self.select_relevant_schema(intent)
            trace.append(f"  → tables: {list(relevant_schema.keys())}")

            trace.append("Step 3: Building query plan")
            plan = self.build_query_plan(
                question,
                intent,
                relevant_schema,
                schema_context=schema_context,
            )
            trace.append(f"  → tables={plan.tables}, agg={plan.aggregation}, group_by={plan.group_by}")

            # Step 4 calls the LLM directly, so it can raise.
            trace.append("Step 4: Generating SQL from plan")
            try:
                sql = self.generate_sql_from_plan(
                    question,
                    intent,
                    plan,
                    relevant_schema,
                    schema_context=schema_context,
                )
                trace.append(f"  → {len(sql)} chars")
            except Exception as e:
                msg = f"SQL generation failed: {e}"
                trace.append(f"  → {msg}")
                return failed(msg)

        # Step 5
        trace.append("Step 5: Validating SQL")
        is_valid, sql, val_warn = self.validate_sql(sql, plan)
        warn.extend(val_warn)
        if not is_valid:
            trace.append(f"  → blocked: {val_warn}")
            return failed(val_warn[0] if val_warn else "SQL validation failed.", sql)
        trace.append(f"  → valid{': ' + '; '.join(val_warn) if val_warn else ''}")

        trace.append("Step 5b: Checking SQL semantics")
        try:
            semantic_warn = self.check_sql_semantics(sql, plan, relevant_schema)
            warn.extend(semantic_warn)
            if semantic_warn:
                trace.append(f"  → {len(semantic_warn)} warning(s)")
            else:
                trace.append("  → no deterministic issues found")
        except Exception as e:
            trace.append(f"  → checker skipped ({e})")

        # Step 6: execute + repair
        trace.append("Step 6: Executing SQL (up to 3 attempts)")
        results: Optional[pd.DataFrame] = None
        last_error: Optional[str] = None

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
                            question,
                            intent,
                            plan,
                            sql,
                            last_error,
                            relevant_schema,
                            schema_context=schema_context,
                        )
                        ok, repaired, rep_warn = self.validate_sql(repaired, plan)
                        warn.extend(rep_warn)
                        if ok:
                            sql = repaired
                            repair_attempts += 1
                            trace.append("  → repaired SQL accepted")
                        else:
                            trace.append("  → repaired SQL failed validation; stopping")
                            break
                    except Exception as rep_e:
                        trace.append(f"  → repair call failed: {rep_e}")
                        break

        if results is None:
            return failed(last_error, sql)

        # Step 7a: Result analysis (pure Python — no LLM call)
        trace.append("Step 7a: Analysing result for suspicious patterns")
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
                    msg += f"Similar values found in the data: {_quoted_list(flags.similar_values)}."
                else:
                    msg += "Consider checking the exact spelling of the entity name."
                warn.append(msg)
            else:
                warn.append("The query returned no rows for the current filter.")

        if flags.suspicious_zero_result:
            if flags.possible_exact_match_miss:
                msg = (
                    f'The result is 0 or null, and the filter used an exact match '
                    f'on "{flags.filter_column}" = "{flags.filter_value}". '
                    "This may not capture the intended entity."
                )
                if flags.similar_values:
                    msg += (
                        f" Similar values in the data include: "
                        f"{_quoted_list(flags.similar_values)}."
                    )
            else:
                msg = (
                    "The aggregate result is 0 or null. "
                    "This may reflect a true zero, or the filter may not have "
                    "matched the intended rows."
                )
            warn.append(msg)

        if flags.entity_match_uncertain and not flags.suspicious_zero_result:
            # Non-zero result but entity matching is uncertain
            warn.append(
                f'The result is based on an exact match for '
                f'"{flags.filter_value}" on column "{flags.filter_column}". '
                f"Other similar values exist ({_quoted_list(flags.similar_values)}); "
                "verify this is the intended entity."
            )

        # Step 7b: Answer summary (the LLM call inside falls back on its own)
        trace.append("Step 7b: Generating answer summary")
        answer_summary = self._generate_answer_summary(question, sql, results, flags, plan)
        trace.append("  → done")

        self.add_to_history(question, {
            'sql_query': sql, 'results': results, 'success': True
        })

        return QueryResponse(
            question=question, interpreted_intent=intent.raw,
            relevant_tables=list(relevant_schema.keys()), query_plan=asdict(plan),
            sql_query=sql, results=results, answer_summary=answer_summary,
            warnings=warn, trace=trace, success=True, repair_attempts=repair_attempts,
            result_scope=self.result_scope_facts(results, plan),
            value_suggestions=(
                {
                    "column": flags.filter_column,
                    "value": flags.filter_value,
                    "alternatives": list(flags.similar_values),
                }
                if flags.similar_values else {}
            ),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute_query_with_explanation(
            self,
            user_question: str,
            schema_context: Optional[str] = None,
            skip_clarification: bool = False,
    ) -> Dict:
        """
        Step 0 — check_clarification_needed:
          If ambiguous → return clarification dict immediately (no SQL generated).
        Steps 1–7 — run_query:
          If clear → run full pipeline and return result dict.

        Always returns a plain dict for backward compatibility with app.py.
        schema_context is optional app/orchestrator routing context. It is kept
        separate from user_question so history, clarification, and UI display
        stay grounded in the user's actual words.
        skip_clarification lets the orchestrator bypass the clarification LLM
        call when it has already determined the route is confident/unambiguous.
        """
        # ── Step 0: Clarification check ──────────────────────────────────
        # check_clarification_needed returns None on any LLM/parse failure, so a
        # broken clarification step degrades into running the query directly.
        clarification = (
            None if skip_clarification
            else self.check_clarification_needed(user_question)
        )

        if clarification is not None:
            return {
                'question': user_question,
                'success': False,
                'needs_clarification': True,
                'clarification_question': clarification.clarification_question,
                'clarification_reason': clarification.clarification_reason,
                'clarification_options': clarification.clarification_options,
                'option_labels': clarification.option_labels,
                'secondary_note': clarification.secondary_note,
                'sql_query': None,
                'results': None,
                'warnings': [],
                'answer_summary': '',
                'trace': [
                    f"Clarification requested ({clarification.ambiguity_type}): "
                    f"'{clarification.ambiguous_term}'"
                ],
                'error': None,
                'repair_attempts': 0,
                'interpreted_intent': {},
                'relevant_tables': [],
                'query_plan': {},
            }

        # ── Steps 1–7: Full pipeline ──────────────────────────────────────
        r = self.run_query(user_question, schema_context=schema_context)
        return {
            'question': r.question,
            'sql_query': r.sql_query,
            'results': r.results,
            'success': r.success,
            'error': r.error,
            'answer_summary': r.answer_summary,
            'warnings': r.warnings,
            'trace': r.trace,
            'interpreted_intent': r.interpreted_intent,
            'relevant_tables': r.relevant_tables,
            'query_plan': r.query_plan,
            'repair_attempts': r.repair_attempts,
            'result_scope': r.result_scope,
            'value_suggestions': r.value_suggestions,
            'needs_clarification': False,
        }

    # ------------------------------------------------------------------
    # Legacy helpers
    # ------------------------------------------------------------------

    def generate_schema_description(self) -> str:
        return self._format_schema_for_prompt(self.schema_info, include_samples=False)

    def get_sample_data_summary(self) -> str:
        return self._format_schema_for_prompt(self.schema_info, include_samples=True)


@dataclass
class AgentOrchestratorConfig:
    deployment_name: str
    memory_limit: int = 8
    prompt_memory_limit: int = 3
    verified_examples_limit: int = 20
    prompt_example_limit: int = 2
    max_routed_tables: int = 3
    max_routed_columns: int = 14
    expanded_routed_tables: int = 5
    expanded_routed_columns: int = 18
    # Skip the clarification LLM call when the schema router is confident and
    # unambiguous (saves one round-trip on clearly-routed questions). This only
    # gates *table* ambiguity; set False to always run the clarification check,
    # which also catches value/metric ambiguity the router cannot see.
    gate_clarification_on_confidence: bool = True
    clarification_skip_confidence: str = "high"  # minimum confidence to skip
    # Deterministic pre-SQL check: when the figure the question names exists
    # under several units / currencies / segments. Independent of the
    # confidence gate above, which only sees table-level ambiguity.
    detect_scope_ambiguity: bool = True
    # How to handle those axes:
    #   "group" — force them into SELECT/GROUP BY so every row carries its own
    #             unit and nothing incompatible is ever summed together. The
    #             user sees all scopes at once and never has to answer anything.
    #   "ask"   — block with a clarification question per axis instead.
    # Grouping is the default: it is as safe as asking but costs no round trip,
    # and the answer can read the unit straight off a result column.
    scope_mode: str = "group"
    # Cap on grouping columns. open_axes is ordered most-dangerous-first
    # (unit, currency, value_kind, ...), so the cap keeps the distinctions that
    # matter and drops the ones that would only multiply rows.
    max_scope_group_columns: int = 4


def estimate_prompt_tokens(text: str) -> int:
    """Cheap conservative-ish estimate for prompt observability."""
    if not text:
        return 0
    return max(1, (len(str(text)) + 3) // 4)


def _append_trace(state: dict, message: str) -> dict:
    trace = list(state.get("graph_trace") or [])
    trace.append(message)
    state["graph_trace"] = trace
    return state


def _truncate(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _result_row_count(result: dict) -> int:
    df = result.get("results") if isinstance(result, dict) else None
    try:
        return int(len(df)) if df is not None else 0
    except Exception:
        return 0


def compact_result_memory(result: dict) -> dict:
    """Keep only small, non-DataFrame details for follow-up context."""
    if not isinstance(result, dict):
        return {}
    return {
        "question": _truncate(result.get("question", ""), 180),
        "success": bool(result.get("success")),
        "needs_clarification": bool(result.get("needs_clarification")),
        "sql_query": _truncate(result.get("sql_query", ""), 500),
        "answer_summary": _truncate(result.get("answer_summary", ""), 240),
        "relevant_tables": list(result.get("relevant_tables") or [])[:5],
        "row_count": _result_row_count(result),
        "warning_count": len(result.get("warnings") or []),
    }


def compact_verified_example(result: dict) -> dict:
    """Store successful question-SQL pairs as retrievable session examples."""
    if not isinstance(result, dict):
        return {}
    sql_query = str(result.get("sql_query") or "").strip()
    if not sql_query:
        return {}
    if not re.match(r"^(SELECT|WITH)\b", sql_query, re.IGNORECASE):
        return {}
    if not result.get("success") or result.get("needs_clarification"):
        return {}
    return {
        "question": _truncate(result.get("question", ""), 180),
        "sql_query": _truncate(sql_query, 700),
        "answer_summary": _truncate(result.get("answer_summary", ""), 220),
        "relevant_tables": list(result.get("relevant_tables") or [])[:5],
        "row_count": _result_row_count(result),
    }


def format_memory_context(memory_items: list[dict], limit: int = 3) -> str:
    """Render bounded memory context for follow-up query planning."""
    recent = [item for item in memory_items if isinstance(item, dict)][-limit:]
    if not recent:
        return ""

    lines = [
        "RECENT QUERY MEMORY:",
        "Use this only to resolve follow-up wording. Do not copy old SQL blindly.",
    ]
    for idx, item in enumerate(recent, 1):
        tables = ", ".join(str(t) for t in item.get("relevant_tables", [])[:3])
        lines.append(
            f'{idx}. Q="{_truncate(item.get("question", ""), 140)}"; '
            f"success={bool(item.get('success'))}; rows={item.get('row_count', 0)}"
            + (f"; tables={tables}" if tables else "")
        )
        if item.get("sql_query"):
            lines.append(f'   SQL="{_truncate(item.get("sql_query", ""), 220)}"')
        if item.get("answer_summary"):
            lines.append(f'   Answer="{_truncate(item.get("answer_summary", ""), 160)}"')
    return "\n".join(lines)


def select_verified_examples(
        examples: list[dict],
        question: str,
        selected_tables: list[str],
        limit: int = 2,
) -> list[dict]:
    """Retrieve compact successful examples relevant to the current question."""
    if not examples:
        return []
    q_terms = schema_terms(question)
    selected_set = {str(t) for t in selected_tables}
    ranked = []
    for example in examples:
        ex_terms = schema_terms(example.get("question", ""))
        ex_tables = {str(t) for t in example.get("relevant_tables", [])}
        score = 3 * len(q_terms & ex_terms) + 5 * len(selected_set & ex_tables)
        if score <= 0:
            continue
        ranked.append((score, example))
    ranked.sort(key=lambda item: (-item[0], str(item[1].get("question", ""))))
    return [example for _, example in ranked[:limit]]


def format_verified_examples_context(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = [
        "VERIFIED QUERY EXAMPLES:",
        "Use these only as style/schema hints for similar questions.",
    ]
    for idx, item in enumerate(examples, 1):
        tables = ", ".join(str(t) for t in item.get("relevant_tables", [])[:3])
        lines.append(
            f'{idx}. Q="{_truncate(item.get("question", ""), 140)}"'
            + (f"; tables={tables}" if tables else "")
        )
        lines.append(f'   SQL="{_truncate(item.get("sql_query", ""), 280)}"')
    return "\n".join(lines)


def analyze_result_sanity(result: dict) -> dict:
    """Summarize result health for graph-level diagnostics."""
    if not isinstance(result, dict):
        return {"status": "missing_result", "flags": ["missing_result"]}

    flags: list[str] = []
    notes: list[str] = []
    df = result.get("results")

    if result.get("needs_clarification"):
        flags.append("needs_clarification")
    if not result.get("success"):
        flags.append("query_failed")
        if result.get("error"):
            notes.append(_truncate(result.get("error"), 180))

    row_count = 0
    col_count = 0
    if isinstance(df, pd.DataFrame):
        row_count = int(len(df))
        col_count = int(len(df.columns))
        if row_count == 0:
            flags.append("empty_result")
        if row_count == 1 and col_count == 1:
            value = df.iloc[0, 0]
            try:
                is_null = bool(pd.isna(value))
            except Exception:
                is_null = value is None
            if is_null:
                flags.append("single_null_value")
            elif isinstance(value, (int, float)) and float(value) == 0.0:
                flags.append("single_zero_value")
        if col_count >= 25:
            flags.append("wide_result")
    elif result.get("success"):
        flags.append("missing_dataframe")

    warnings_count = len(result.get("warnings") or [])
    if warnings_count:
        flags.append("has_warnings")

    status = "ok"
    if any(flag in flags for flag in ("query_failed", "needs_clarification", "empty_result")):
        status = "needs_attention"
    elif flags:
        status = "review"

    return {
        "status": status,
        "flags": list(dict.fromkeys(flags)),
        "notes": notes[:3],
        "row_count": row_count,
        "column_count": col_count,
        "warning_count": warnings_count,
    }


def _coerce_sqlite_value(value):
    """Normalize values that SQLite cannot bind directly from pandas objects."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _ensure_connection(files_db: FilesDatabaseManager) -> None:
    files_db.ensure_connection()


def _sqlite_ready_frame(files_db: FilesDatabaseManager, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Copy `df` with cleaned column names and SQLite-bindable values.

    Returns None when there is nothing to load.
    """
    working_df = cast(pd.DataFrame, pd.DataFrame(df).copy())
    if working_df.empty:
        return None
    working_df.attrs = {}

    working_df.columns = [
        str(name) for name in files_db._clean_column_names(list(working_df.columns))
    ]

    for col in working_df.columns:
        series = working_df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            dt = pd.to_datetime(series, errors="coerce")
            working_df[col] = dt.dt.strftime("%Y-%m-%d %H:%M:%S").where(
                ~dt.isna(),
                None,
            )
            continue
        if series.dtype == object:
            working_df[col] = series.map(_coerce_sqlite_value)

    return working_df


def _store_frame_in_files_db(
        files_db: FilesDatabaseManager,
        table_name: str,
        working_df: pd.DataFrame,
        source_file_name: str,
        source_sheet_name: str,
) -> str:
    """Write a prepared frame to SQLite, record its schema, and cache the frame."""
    working_df.to_sql(table_name, files_db.connection, if_exists="replace", index=False)
    files_db.tables_info[table_name] = {
        "source_file": source_file_name,
        "source_sheet": source_sheet_name,
        "columns": list(working_df.columns),
        "row_count": int(working_df.shape[0]),
        "column_types": {str(col): str(dtype) for col, dtype in working_df.dtypes.items()},
    }

    frame_cache = getattr(files_db, "flat_file_frames", None)
    if not isinstance(frame_cache, dict):
        frame_cache = {}
        files_db.flat_file_frames = frame_cache
    frame_cache[table_name] = working_df.copy()
    return table_name


def load_dataframe_into_files_db(
        files_db: FilesDatabaseManager,
        df: pd.DataFrame,
        source_file_name: str,
        source_sheet_name: str,
) -> str:
    """Load an in-memory dataframe as a table inside FilesDatabaseManager."""
    files_db.ensure_connection()
    working_df = _sqlite_ready_frame(files_db, df)
    if working_df is None:
        return ""

    table_name = str(files_db._clean_table_name(source_sheet_name or "sheet"))
    # A same-named sheet from a different workbook must not overwrite the first.
    existing = files_db.tables_info.get(table_name)
    if existing and (
            str(existing.get("source_file", "")) != str(source_file_name)
            or str(existing.get("source_sheet", "")) != str(source_sheet_name)
    ):
        file_prefix = Path(source_file_name).stem or "file"
        table_name = str(
            files_db._clean_table_name(f"{file_prefix}_{source_sheet_name or 'sheet'}")
        )
        suffix = 2
        candidate = table_name
        while candidate in files_db.tables_info:
            candidate = str(files_db._clean_table_name(f"{table_name}_{suffix}"))
            suffix += 1
        table_name = candidate
    if not table_name:
        return ""

    return _store_frame_in_files_db(
        files_db, table_name, working_df, source_file_name, source_sheet_name
    )


def replace_table_in_files_db(
        files_db: FilesDatabaseManager,
        table_name: str,
        df: pd.DataFrame,
        source_file_name: str,
        source_sheet_name: str,
) -> str:
    """Replace an existing in-memory table with cleaned dataframe content."""
    files_db.ensure_connection()
    working_df = _sqlite_ready_frame(files_db, df)
    if working_df is None:
        return ""

    target_table = str(files_db._clean_table_name(table_name))
    if not target_table:
        return ""

    return _store_frame_in_files_db(
        files_db, target_table, working_df, source_file_name, source_sheet_name
    )


def refresh_flat_file_schema_artifacts(files_db: FilesDatabaseManager) -> None:
    """Synchronize agent schema context and the downloadable common workbook."""
    if files_db is None or not files_db.tables_info:
        return

    for stale_name in [
        name for name in list(files_db.tables_info) if str(name).lower() == "schema"
    ]:
        try:
            if files_db.connection is not None:
                files_db.connection.exec_driver_sql(f'DROP TABLE IF EXISTS "{stale_name}"')
        except Exception:
            pass
        files_db.tables_info.pop(stale_name, None)
        stale_cache = getattr(files_db, "flat_file_frames", None)
        if isinstance(stale_cache, dict):
            stale_cache.pop(stale_name, None)

    sources: list[dict] = []
    workbook_frames: dict[str, pd.DataFrame] = {}
    frame_cache = getattr(files_db, "flat_file_frames", None)
    if not isinstance(frame_cache, dict):
        frame_cache = {}
        files_db.flat_file_frames = frame_cache

    for table_name, info in list(files_db.tables_info.items()):
        if str(table_name).lower() == "schema":
            continue
        cached_frame = frame_cache.get(str(table_name))
        if isinstance(cached_frame, pd.DataFrame):
            frame = cached_frame.copy()
        else:
            try:
                frame = pd.DataFrame(files_db.execute_query(f'SELECT * FROM "{table_name}"'))
                frame_cache[str(table_name)] = frame.copy()
            except Exception as exc:
                print(f"[schema_sync] Could not read table '{table_name}': {exc}")
                continue
        if frame.empty:
            continue

        source_file = str(info.get("source_file") or "uploaded_file.xlsx")
        source_sheet = str(info.get("source_sheet") or table_name)
        sources.append(
            {
                "file_name": source_file,
                "sheet_name": source_sheet,
                "physical_table_name": str(table_name),
                "sheet_index": len(sources),
                "parsing_mode": "flat_file_builder",
                "source_shape": frame.shape,
                "frame": frame,
            }
        )
        workbook_frames[str(table_name)] = frame

    if not sources:
        return

    package = build_excel_schema_package(sources)
    schema_frame = build_embedded_schema_frame(package)
    if schema_frame.empty:
        return

    files_db.flat_file_schema_catalog = package["catalog"]
    files_db.flat_file_schema_frame = schema_frame
    files_db.schema_router_index = build_schema_router_index(
        files_db,
        package,
        workbook_frames,
    )
    # The downloadable common workbook is NOT built here: the openpyxl write
    # is the single most expensive post-load step (it can exceed extraction
    # itself), and most sessions never download it. It is produced on demand
    # by build_common_workbook_bytes; clearing the cache here keeps any
    # previously prepared download from going stale after tables change.
    files_db.common_workbook_bytes = None


def build_common_workbook_bytes(files_db: FilesDatabaseManager) -> bytes:
    """Build (and cache) the downloadable common workbook on demand.

    Kept out of the upload path deliberately — writing every loaded table into
    one xlsx via openpyxl is the most expensive post-load step. The result is
    cached on files_db; refresh_flat_file_schema_artifacts clears that cache
    whenever the loaded tables change.
    """
    if files_db is None or not files_db.tables_info:
        return b""
    cached = getattr(files_db, "common_workbook_bytes", None)
    if cached:
        return cached

    frame_cache = getattr(files_db, "flat_file_frames", None)
    if not isinstance(frame_cache, dict):
        frame_cache = {}
    workbook_frames: Dict[str, pd.DataFrame] = {}
    for table_name in files_db.tables_info:
        if str(table_name).lower() == "schema":
            continue
        frame = frame_cache.get(str(table_name))
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            try:
                frame = pd.DataFrame(
                    files_db.execute_query(f'SELECT * FROM "{table_name}"')
                )
            except Exception:
                continue
        if not frame.empty:
            workbook_frames[str(table_name)] = frame
    if not workbook_frames:
        return b""

    schema_frame = getattr(files_db, "flat_file_schema_frame", None)
    schema_df = (
        schema_frame
        if isinstance(schema_frame, pd.DataFrame) and not schema_frame.empty
        else None
    )
    payload = to_multisheet_excel_bytes(workbook_frames, schema_df=schema_df)
    files_db.common_workbook_bytes = payload
    return payload


# (name fragment, phrase) pairs used to describe a table when no LLM is
# available. Ordered most-specific first; the first match wins per category.
_TABLE_DOMAIN_HINTS: List[Tuple[str, str]] = [
    ("balance sheet", "balance-sheet positions"),
    ("financial position", "balance-sheet positions"),
    ("cash flow", "cash-flow movements"),
    ("income statement", "income-statement lines"),
    ("profit and loss", "profit-and-loss lines"),
    ("shareholders", "shareholders' equity movements"),
    ("sh equity", "shareholders' equity movements"),
    ("solvency", "solvency and capital-adequacy metrics"),
    ("asset alloc", "asset-allocation breakdown"),
    ("new business", "new-business figures"),
    ("market data", "market rates and reference data"),
    ("aum", "assets-under-management figures"),
    ("consolidation", "consolidation adjustments"),
    ("property-casualty", "property-casualty results"),
    ("life health", "life & health results"),
    ("asset management", "asset-management results"),
    ("corporate", "corporate-segment results"),
    ("index", "an index of the workbook contents"),
    ("cover", "cover-page information"),
]

_TABLE_BREAKDOWN_HINTS: List[Tuple[str, str]] = [
    ("cust segment", "by customer segment"),
    ("segment", "by segment"),
    ("by region", "by region"),
    ("region", "by region"),
    ("by country", "by country"),
]


def heuristic_table_summary(table_name: str, info: Dict) -> str:
    """Deterministic one-line description of a table from its name + schema.

    Needs no LLM and issues no queries, so it always renders. The AI path in
    FilesSQLAgent.summarize_tables refines this into content-aware wording.
    """
    # Match against both the raw name and an underscore-normalized form, since
    # physical table names are cleaned to snake_case ("balance_sheet") while the
    # source sheet name ("Balance Sheet") keeps spaces.
    source_sheet = str(info.get("source_sheet", "") or "")
    lname = " ".join(
        f"{table_name} {source_sheet}".lower().replace("_", " ").split()
    )
    columns = [str(column) for column in info.get("columns", [])]
    column_set = {column.lower() for column in columns}
    row_count = int(info.get("row_count", 0) or 0)

    domain = next(
        (phrase for fragment, phrase in _TABLE_DOMAIN_HINTS if fragment in lname),
        "",
    )
    breakdown = next(
        (phrase for fragment, phrase in _TABLE_BREAKDOWN_HINTS if fragment in lname),
        "",
    )
    if "line_item" in column_set and not domain:
        domain = "financial line items"
    if not domain:
        domain = "tabular records"

    period = ""
    if "ytd" in lname:
        period = ", year-to-date"
    elif "qtd" in lname:
        period = ", quarter-to-date"

    dimensions = [
        label
        for column, label in (
            ("period", "period"),
            ("valuation_date", "date"),
            ("unit", "unit"),
            ("currency", "currency"),
            ("section", "section"),
            ("column_group", "group"),
        )
        if column in column_set
    ]
    dim_clause = f" broken down by {', '.join(dimensions[:3])}" if dimensions else ""

    summary = f"{domain.capitalize()}{breakdown_prefix(breakdown)}{period}{dim_clause}."
    summary += f" {row_count:,} row{'s' if row_count != 1 else ''}, {len(columns)} columns."
    return summary


def breakdown_prefix(breakdown: str) -> str:
    return f" {breakdown}" if breakdown else ""


def summarize_loaded_tables(
        files_db: FilesDatabaseManager,
) -> Dict[str, Dict[str, str]]:
    """Heuristic-only overviews for every loaded table (no LLM needed)."""
    return {
        table_name: {
            "summary": heuristic_table_summary(table_name, info),
            "source": "heuristic",
        }
        for table_name, info in (files_db.tables_info or {}).items()
    }


def schema_terms(value: Any) -> set[str]:
    """Normalize schema and question text into retrieval terms."""
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "show",
        "the",
        "this",
        "to",
        "was",
        "what",
        "which",
        "with",
    }
    tokens = {
        token for token in text.split() if len(token) >= 2 and token not in stopwords
    }
    aliases = {
        "sales": {"revenue", "turnover"},
        "revenue": {"sales", "turnover"},
        "cash": {"liquidity"},
        "liquidity": {"cash"},
        "debt": {"borrowings", "liabilities"},
        "borrowings": {"debt"},
        "profit": {"income", "earnings"},
        "income": {"profit", "earnings"},
        "customer": {"client"},
        "customers": {"customer", "client"},
        "segment": {"business", "division"},
        "year": {"period", "date"},
        "quarter": {"period", "date"},
    }
    expanded = set(tokens)
    for token in tokens:
        expanded.update(aliases.get(token, set()))
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
    return expanded


def infer_router_column_role(column: str, dtype: str) -> str:
    normalized = column.lower()
    if normalized in {
        "value_numeric",
        "amount",
        "balance",
        "revenue",
        "cost",
        "quantity",
        "count",
    } or any(
        token in normalized for token in ("amount", "value", "balance", "revenue", "cost")
    ):
        return "measure"
    if any(token in normalized for token in ("date", "period", "year", "quarter", "month")):
        return "time_dimension"
    if normalized in {"line_item", "line_item_path", "parent_line_item", "section"}:
        return "financial_hierarchy"
    if any(token in dtype.lower() for token in ("int", "float", "decimal")):
        return "numeric_attribute"
    return "dimension"


def build_schema_router_index(
        files_db: FilesDatabaseManager,
        schema_package: dict,
        workbook_frames: dict[str, pd.DataFrame],
) -> list[dict]:
    """Build a local retrieval index over physical SQLite tables."""
    catalog = schema_package.get("catalog", {}) if schema_package else {}
    logical_tables = catalog.get("logical_tables", [])

    # Join on the physical table name, which is unique per loaded table.
    # Joining on the sheet name let two workbooks that share a tab title
    # (both having a "Summary") inherit each other's semantics and search
    # terms, which then mis-routed questions between unrelated files.
    logical_by_table: dict[str, list[dict]] = {}
    for logical in logical_tables:
        physical = str(logical.get("physical_table") or "").strip().lower()
        if physical:
            logical_by_table.setdefault(physical, []).append(logical)

    index: list[dict] = []
    for table_name, info in files_db.tables_info.items():
        if str(table_name).lower() == "schema":
            continue
        frame = workbook_frames.get(str(table_name), pd.DataFrame())
        columns = [str(column) for column in info.get("columns", [])]
        frame_values = frame.to_numpy(dtype=object, copy=False) if not frame.empty else None
        frame_column_positions = {
            str(column): idx for idx, column in enumerate(frame.columns)
        }
        column_types = {
            str(column): str(dtype) for column, dtype in info.get("column_types", {}).items()
        }
        source_sheet = str(info.get("source_sheet") or table_name)
        related_logical = logical_by_table.get(str(table_name).strip().lower(), [])
        semantic_terms: set[str] = set()
        table_types: set[str] = set()
        aggregation_rules: set[str] = set()
        # Aggregation guardrails the catalog derived at load time. Without these
        # the model sees a measure column and no reason not to SUM it.
        primary_measures: set[str] = set()
        known_units: set[str] = set()
        non_additive_dimensions: set[str] = set()
        requires_unit_filter = False
        grain = ""
        for logical in related_logical:
            semantic_terms.update(schema_terms(logical.get("display_name", "")))
            semantic_terms.update(schema_terms(logical.get("description", "")))
            semantic_terms.update(schema_terms(" ".join(logical.get("search_terms", []))))
            table_types.add(str(logical.get("table_type") or ""))
            aggregation_rules.add(
                f"{logical.get('default_aggregation', 'NONE')}/"
                f"{logical.get('additivity', 'unknown')}"
            )
            if logical.get("primary_measure"):
                primary_measures.add(str(logical["primary_measure"]))
            known_units.update(str(u) for u in (logical.get("known_units") or []) if str(u).strip())
            non_additive_dimensions.update(
                str(d) for d in (logical.get("non_additive_dimensions") or []) if str(d).strip()
            )
            requires_unit_filter = requires_unit_filter or bool(
                int(logical.get("requires_unit_filter") or 0)
            )
            grain = grain or str(logical.get("grain") or "")

        column_records = []
        for column in columns:
            sample_values: list[str] = []
            seen_samples: set[str] = set()
            column_idx = frame_column_positions.get(column)
            if frame_values is not None and column_idx is not None:
                for value in frame_values[:, column_idx]:
                    try:
                        if pd.isna(value):
                            continue
                    except Exception:
                        pass
                    rendered = str(value).strip()
                    if (
                            rendered
                            and rendered.lower() not in {"nan", "none"}
                            and rendered not in seen_samples
                    ):
                        sample_values.append(rendered[:80])
                        seen_samples.add(rendered)
                    if len(sample_values) >= 5:
                        break
            column_records.append(
                {
                    "name": column,
                    "dtype": column_types.get(column, "unknown"),
                    "role": infer_router_column_role(column, column_types.get(column, "")),
                    "samples": sample_values,
                    "name_terms": sorted(schema_terms(column)),
                    "sample_terms": sorted(schema_terms(" ".join(sample_values))),
                }
            )

        index.append(
            {
                "table_name": str(table_name),
                "source_file": str(info.get("source_file") or ""),
                "source_sheet": source_sheet,
                "row_count": int(info.get("row_count", 0) or 0),
                "table_types": sorted(value for value in table_types if value),
                "aggregation_rules": sorted(aggregation_rules),
                "primary_measures": sorted(primary_measures),
                "known_units": sorted(known_units),
                "non_additive_dimensions": sorted(non_additive_dimensions),
                "requires_unit_filter": requires_unit_filter,
                "grain": grain,
                "columns": column_records,
                "terms": sorted(
                    schema_terms(table_name)
                    | schema_terms(source_sheet)
                    | schema_terms(info.get("source_file", ""))
                    | semantic_terms
                ),
            }
        )
    return index


def route_schema_for_question(
        files_db: FilesDatabaseManager,
        user_question: str,
        max_tables: int = 3,
        max_columns: int = 14,
) -> dict:
    """Select only schema fragments relevant to one user question."""
    index = list(getattr(files_db, "schema_router_index", []) or [])
    question_terms = schema_terms(user_question)
    question_text = re.sub(r"[^a-z0-9]+", " ", str(user_question or "").lower()).strip()
    ranked = []
    for table in index:
        score = 4.0 * len(question_terms & set(table.get("terms", [])))
        for name in (str(table.get("table_name", "")), str(table.get("source_sheet", ""))):
            normalized_name = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
            if normalized_name and normalized_name in question_text:
                score += 12.0

        column_scores = []
        for column in table.get("columns", []):
            column_score = (
                    5.0 * len(question_terms & set(column.get("name_terms", [])))
                    + 1.5 * len(question_terms & set(column.get("sample_terms", [])))
            )
            normalized_column = re.sub(
                r"[^a-z0-9]+",
                " ",
                str(column.get("name", "")).lower(),
            ).strip()
            if normalized_column and normalized_column in question_text:
                column_score += 10.0
            if column.get("role") == "measure" and question_terms & {
                "sum",
                "total",
                "average",
                "avg",
                "amount",
                "value",
            }:
                column_score += 1.5
            column_scores.append((column_score, column))
            score += min(column_score, 8.0) * 0.35
        column_scores.sort(key=lambda item: (-item[0], str(item[1].get("name", ""))))
        ranked.append(
            {
                "score": round(score, 3),
                "table": table,
                "columns": [
                    column for column_score, column in column_scores if column_score > 0
                ][:max_columns],
            }
        )

    ranked.sort(key=lambda item: (-item["score"], str(item["table"].get("table_name", ""))))
    top_score = float(ranked[0]["score"]) if ranked else 0.0
    relevance_floor = max(3.0, top_score * 0.35)
    positive = [item for item in ranked if item["score"] >= relevance_floor]
    selected = (positive or ranked)[:max_tables]
    for item in selected:
        if not item["columns"]:
            item["columns"] = item["table"].get("columns", [])[:max_columns]
            continue
        selected_names = {str(column.get("name", "")) for column in item["columns"]}
        support_roles = {"measure", "time_dimension", "financial_hierarchy"}
        for column in item["table"].get("columns", []):
            if len(item["columns"]) >= max_columns:
                break
            if (
                    column.get("role") in support_roles
                    and str(column.get("name", "")) not in selected_names
            ):
                item["columns"].append(column)
                selected_names.add(str(column.get("name", "")))

    ambiguous = False
    second_score = 0.0
    if len(selected) >= 2:
        top_score = float(selected[0]["score"])
        second_score = float(selected[1]["score"])
        ambiguous = top_score <= 0 or second_score >= max(top_score * 0.80, top_score - 3.0)
    score_gap = round(float(top_score - second_score), 3)
    if not selected or top_score <= 0:
        confidence = "low"
    elif ambiguous:
        confidence = "medium"
    elif top_score >= 8 or score_gap >= 5:
        confidence = "high"
    else:
        confidence = "medium"
    return {
        "selected": selected,
        "ambiguous": ambiguous,
        "available_table_count": len(index),
        "top_score": round(float(top_score), 3),
        "second_score": round(float(second_score), 3),
        "score_gap": score_gap,
        "confidence": confidence,
        "max_tables": max_tables,
        "max_columns": max_columns,
    }


def format_routed_schema_context(route: dict) -> str:
    """Render a small SQL contract for selected schema fragments."""
    lines = [
        "SQL DATA CONTRACT:",
        "Schema metadata was routed locally; do not inspect metadata tables.",
        "Use only the candidate business tables and columns listed below.",
    ]
    selected = route.get("selected", [])
    for item in selected:
        table = item["table"]
        lines.append(
            f'TABLE "{table["table_name"]}" '
            f'({table.get("row_count", 0)} rows; '
            f'source {table.get("source_file", "")} / '
            f'{table.get("source_sheet", "")})'
        )
        if table.get("table_types"):
            lines.append("  Semantic type: " + ", ".join(table["table_types"]))
        if table.get("aggregation_rules"):
            lines.append("  Aggregation policy: " + ", ".join(table["aggregation_rules"]))
        if table.get("grain"):
            lines.append(f"  Grain: {table['grain']}")
        if table.get("primary_measures"):
            lines.append("  Primary measure: " + ", ".join(table["primary_measures"]))
        # The two rules that stop a plausible-looking but meaningless aggregate.
        if table.get("requires_unit_filter"):
            units = ", ".join(table.get("known_units") or []) or "several"
            lines.append(
                f"  ⚠ MUST filter or group by unit before aggregating — this table "
                f"mixes units ({units}); summing across them is invalid."
            )
        elif table.get("known_units"):
            lines.append("  Units present: " + ", ".join(table["known_units"]))
        if table.get("non_additive_dimensions"):
            dims = ", ".join(table["non_additive_dimensions"])
            lines.append(
                f"  ⚠ NEVER SUM across these dimensions (values are ratios or "
                f"point-in-time balances): {dims}. Use AVG or filter to one value."
            )
        for column in item.get("columns", []):
            sample_text = (
                f"; examples={json.dumps(column.get('samples', [])[:3], ensure_ascii=False)}"
                if column.get("samples")
                else ""
            )
            lines.append(
                f'  - "{column["name"]}": {column.get("dtype", "unknown")}; '
                f'role={column.get("role", "dimension")}{sample_text}'
            )
    lines.extend(
        [
            "RULES:",
            "- Never query Schema/schema; it is not a database table.",
            "- Quote identifiers with double quotes.",
            "- Use numeric measures for arithmetic and preserve unit/currency scope.",
            "- Do not join candidate tables unless explicitly required.",
        ]
    )
    if route.get("ambiguous"):
        candidate_names = [item["table"]["table_name"] for item in selected]
        lines.append(
            "- Ambiguity warning: several tables match similarly "
            f"({', '.join(candidate_names)}). Ask the user to choose if the "
            "question does not clearly identify one."
        )
    return "\n".join(lines)


# ============================================================================
# Deterministic scope-ambiguity detection
# ============================================================================
#
# The schema router picks *tables*; it cannot see that one line item (say
# "GWP") exists many times inside a table under different units, currencies,
# segments or periods. Aggregating across those axes silently produces a
# meaningless number, so before any SQL runs we look up the real values behind
# the question's subject and ask the user to pin the axes that actually differ.

# Columns that usually hold the figure's *name* — the subject of the question.
_SUBJECT_COLUMN_NAMES = (
    "line_item", "line_item_path", "metric", "metric_detail",
    "label", "item", "kpi", "description", "caption", "name",
)

# Columns whose values define the *scope* of a figure, most dangerous first.
# Mixing unit or currency is always wrong; mixing a segment is merely broad.
_SCOPE_AXES: List[Tuple[str, str]] = [
    ("unit", "unit"),
    ("currency", "currency"),
    ("value_kind", "value kind"),
    ("table_name", "table"),
    ("section", "section"),
    ("column_group", "column group"),
    ("segment", "segment"),
    ("period", "period"),
    ("metric", "metric"),
]

# Words that mean "break the answer down by this axis" rather than "filter to
# one value of it" — asking about a requested breakdown would be wrong.
_AXIS_GROUPING_WORDS = {
    "period": {"period", "periods", "year", "years", "yearly", "annual", "annually",
               "quarter", "quarters", "quarterly", "month", "months", "monthly",
               "time", "fy", "ytd"},
    "unit": {"unit", "units"},
    "currency": {"currency", "currencies", "fx"},
    "section": {"section", "sections"},
    "column_group": {"group", "groups"},
    "segment": {"segment", "segments", "division", "divisions", "business"},
    "table_name": {"table", "tables", "tab", "tabs", "sheet", "sheets"},
    "metric": {"metric", "metrics", "kpi", "kpis"},
    "value_kind": {"kind", "kinds"},
}

_GROUPING_PHRASES = ("per ", "by ", "each ", "for every ", "split by ", "broken down by ",
                     "across ", "over ")

# Labels that mark a pre-aggregated row. Summing these together with the detail
# rows they summarise double-counts, which is why it is a user decision.
_SUBTOTAL_TERMS = ("total", "subtotal", "sub-total", "grand total", "sum of")

# Row-inclusion answers are matched back verbatim from the follow-up question,
# so the wording must be distinctive and stable.
OPT_EXCLUDE_ISOLATED = "Exclude stray helper cells (is_isolated_cell = 0)"
OPT_INCLUDE_ISOLATED = "Include stray helper cells as well"
OPT_DETAIL_ROWS_ONLY = "Detail rows only (exclude subtotal/total rows)"
OPT_TOTAL_ROWS_ONLY = "Pre-aggregated subtotal/total rows only"
OPT_DETAIL_AND_TOTALS = "Both detail and total rows (may double-count)"

# Escape hatch: skip the remaining scope questions and answer immediately. The
# axes left open are reported as a warning so the mixing is never silent.
OPT_ANSWER_ANYWAY = "Answer anyway without narrowing the scope"


# Question phrasing that is never the name of a figure.
_SUBJECT_STOPWORDS = {
    "give", "list", "get", "find", "tell", "display", "want", "need", "please",
    "per", "each", "every", "across", "split", "broken", "down", "between",
    "total", "totals", "sum", "average", "avg", "count", "number", "numbers",
    "value", "values", "figure", "figures", "amount", "amounts", "data", "table",
}


def _question_requests_breakdown(question: str, column: str) -> bool:
    """True when the question asks to break results down by this axis.

    "GWP per year" wants a row per period, so the period axis is a requested
    grouping and must not be turned into a clarification question.
    """
    text = re.sub(r"[^a-z0-9]+", " ", str(question or "").lower())
    axis_words = _AXIS_GROUPING_WORDS.get(column, set()) | {column.replace("_", " ")}
    return any(
        f"{phrase}{word}" in text
        for phrase in _GROUPING_PHRASES
        for word in axis_words
    )


def _question_pins_value(question: str, values: List[str]) -> bool:
    """True when the question already names one of the axis's values."""
    text = str(question or "").lower()
    return any(str(v).strip().lower() in text for v in values if str(v).strip())


def _subject_tokens(question: str) -> List[str]:
    """Candidate words for the figure being asked about."""
    return sorted({
        token for token in schema_terms(question)
        if len(token) >= 3 and not token.isdigit() and token not in _SUBJECT_STOPWORDS
    })


def detect_scope_ambiguity(
        files_db: FilesDatabaseManager,
        user_question: str,
        route: Optional[dict] = None,
        max_tables: int = 3,
        max_options: int = 8,
) -> Optional[dict]:
    """Find the one scope axis the user most needs to pin down, or None.

    Returns a dict describing the subject, the axis to ask about, its real
    values, and any further axes that will still be ambiguous afterwards.
    """
    if files_db is None or not getattr(files_db, "tables_info", None):
        return None
    if files_db.connection is None:
        return None

    quote = FilesSQLAgent._quote_ident
    like = FilesSQLAgent._escape_like_literal
    escape = FilesSQLAgent._LIKE_ESCAPE

    candidate_tables = [
        str(item.get("table", {}).get("table_name", ""))
        for item in ((route or {}).get("selected") or [])
    ]
    candidate_tables = [t for t in candidate_tables if t in files_db.tables_info]
    if not candidate_tables:
        candidate_tables = list(files_db.tables_info)
    candidate_tables = candidate_tables[:max_tables]

    tokens = _subject_tokens(user_question)
    if not tokens:
        return None

    def distinct_values(table: str, column: str, where: str = "", limit: int = 60) -> List[str]:
        q_col = quote(column)
        clause = f"WHERE {where} AND " if where else "WHERE "
        try:
            df = files_db.execute_query(
                f"SELECT DISTINCT TRIM(CAST({q_col} AS TEXT)) AS v FROM {quote(table)} "
                f"{clause}{q_col} IS NOT NULL AND TRIM(CAST({q_col} AS TEXT)) <> '' "
                f"ORDER BY v LIMIT {limit}"
            )
        except Exception:
            return []
        return [
            str(v) for v in df["v"].tolist()
            if str(v).strip() and str(v).lower() not in ("nan", "none", "null")
        ]

    # ── Locate the question's subject in a label-like column ────────────────
    # Every candidate token is tried; the most *selective* one wins (the token
    # matching the fewest distinct labels), so "GWP" beats a vague word like
    # "year" that happens to appear in some label.
    by_token: Dict[str, List[tuple]] = {}
    for token in tokens:
        matches = []
        for table in candidate_tables:
            lower_columns = {
                str(c).lower(): str(c)
                for c in files_db.tables_info[table].get("columns", [])
            }
            for name in _SUBJECT_COLUMN_NAMES:
                column = lower_columns.get(name)
                if not column:
                    continue
                where = (
                    f"LOWER(CAST({quote(column)} AS TEXT)) LIKE '%{like(token.lower())}%' "
                    f"ESCAPE '{escape}'"
                )
                values = distinct_values(table, column, where)
                if values:
                    matches.append((table, column, token, values, where))
                break  # one subject column per table is enough
        if matches:
            by_token[token] = matches

    axes: List[dict] = []
    question_lower = str(user_question or "").lower()
    resolved_filters: List[str] = []

    if by_token:
        def selectivity(item) -> tuple:
            token, matches = item
            distinct_labels = len({v for _, _, _, values, _ in matches for v in values})
            return distinct_labels, -len(token), token

        subject_token, subject = min(by_token.items(), key=selectivity)
        subject_values = sorted({v for _, _, _, values, _ in subject for v in values})

        # ── Axis 0: the subject resolves to several different tables ───────
        if len(subject) > 1:
            axes.append({
                "column": "table_name",
                "label": "table",
                "values": sorted({table for table, _, _, _, _ in subject}),
                "scope": "table",
            })

        table, column, _, _, where = subject[0]
        pinned_subject = sorted(
            (v for v in subject_values if v.strip().lower() in question_lower),
            key=len,
            reverse=True,
        )
        if pinned_subject:
            resolved_filters.append(
                f"TRIM(CAST({quote(column)} AS TEXT)) = "
                f"'{FilesSQLAgent._escape_sql_literal(pinned_subject[0])}'"
            )
        elif len(subject_values) > 1:
            axes.append({
                "column": column,
                "label": column.replace("_", " "),
                "values": subject_values,
                "scope": "subject",
            })
    else:
        # No named figure — a broad question such as "total value by section".
        # Row-inclusion decisions still matter, so keep going over the whole table.
        subject_token, subject_values = "", []
        table, column, where = candidate_tables[0], "", "1=1"

    # ── Remaining axes: scope columns that differ across the matched rows ──
    # Values the question already pins become filters, so each later axis is
    # measured only over the rows still in play. Without this the drill-down
    # keeps asking about distinctions the earlier answers already removed.
    lower_columns = {str(c).lower(): str(c) for c in files_db.tables_info[table].get("columns", [])}

    def scoped_where() -> str:
        return " AND ".join([where] + resolved_filters)

    for axis_column, axis_label in _SCOPE_AXES:
        actual = lower_columns.get(axis_column)
        if not actual or actual == column:
            continue

        values = distinct_values(table, actual, scoped_where())
        if len(values) < 2:
            continue

        # Longest match first so "EUR mn" wins over a bare "EUR".
        pinned = sorted(
            (v for v in values if v.strip().lower() in question_lower),
            key=len,
            reverse=True,
        )
        if pinned:
            resolved_filters.append(
                f"TRIM(CAST({quote(actual)} AS TEXT)) = "
                f"'{FilesSQLAgent._escape_sql_literal(pinned[0])}'"
            )
            continue

        if _question_requests_breakdown(user_question, axis_column):
            continue

        axes.append({
            "column": actual,
            "label": axis_label,
            "values": values,
            "scope": "scope",
        })

    # ── Row-inclusion decisions ────────────────────────────────────────────
    # These are the two guardrail warnings the user could never act on: stray
    # helper cells, and detail rows mixed with the totals that summarise them.
    # Both are offered as an explicit choice that becomes a real SQL filter.
    def add_row_axis(column_name: str, options: List[str],
                     predicates: Dict[str, str], label: str) -> None:
        chosen = [opt for opt in options if opt.lower() in question_lower]
        if chosen:
            predicate = predicates.get(chosen[0], "")
            if predicate:
                resolved_filters.append(predicate)
            return
        axes.append({
            "column": column_name,
            "label": label,
            "values": options,
            "scope": "rows",
        })

    isolated_column = lower_columns.get("is_isolated_cell")
    if isolated_column:
        flags = {v.strip() for v in distinct_values(table, isolated_column, scoped_where())}
        if len(flags) > 1:
            add_row_axis(
                isolated_column,
                [OPT_EXCLUDE_ISOLATED, OPT_INCLUDE_ISOLATED],
                {OPT_EXCLUDE_ISOLATED:
                    f"TRIM(CAST({quote(isolated_column)} AS TEXT)) = '0'"},
                "stray cells",
            )

    label_column = column or next(
        (lower_columns[n] for n in _SUBJECT_COLUMN_NAMES if n in lower_columns), ""
    )
    if label_column:
        labels = distinct_values(table, label_column, scoped_where())
        totals = [v for v in labels if any(t in v.lower() for t in _SUBTOTAL_TERMS)]
        details = [v for v in labels if v not in totals]
        if totals and details:
            q_label = quote(label_column)
            not_total = " AND ".join(
                f"LOWER(CAST({q_label} AS TEXT)) NOT LIKE '%{term}%'"
                for term in _SUBTOTAL_TERMS
            )
            is_total = " OR ".join(
                f"LOWER(CAST({q_label} AS TEXT)) LIKE '%{term}%'"
                for term in _SUBTOTAL_TERMS
            )
            add_row_axis(
                label_column,
                [OPT_DETAIL_ROWS_ONLY, OPT_TOTAL_ROWS_ONLY, OPT_DETAIL_AND_TOTALS],
                {
                    OPT_DETAIL_ROWS_ONLY: not_total,
                    OPT_TOTAL_ROWS_ONLY: f"({is_total})",
                },
                "subtotal rows",
            )

    # The user asked to stop narrowing: answer now, but keep the open axes so
    # the caller can say exactly what got mixed together.
    answer_anyway = OPT_ANSWER_ANYWAY.lower() in question_lower

    base = {
        "table": table,
        "subject_column": column,
        "subject_token": subject_token,
        "subject_values": subject_values,
        "resolved_filters": resolved_filters,
        # Every axis still in play. In grouping mode these become GROUP BY
        # columns; in asking mode the first becomes the question.
        "open_axes": [
            {"column": a["column"], "label": a["label"], "values": a["values"][:max_options]}
            for a in axes
        ],
        "skipped_axes": [
            {"label": a["label"], "values": a["values"][:max_options]} for a in axes
        ] if answer_anyway else [],
    }

    if not axes or answer_anyway:
        return {**base, "axis": None, "options": [], "remaining_axes": []}

    primary = axes[0]
    return {
        **base,
        "axis": primary,
        "options": primary["values"][:max_options],
        "remaining_axes": [a["label"] for a in axes[1:]],
    }


def scope_clarification_result(user_question: str, ambiguity: dict) -> dict:
    """Render a scope ambiguity as the standard clarification result dict."""
    axis = ambiguity["axis"]
    subject = ambiguity["subject_token"].upper()
    options = ambiguity["options"]
    label = axis["label"]

    scope_kind = axis["scope"]
    named = f'"{subject}"' if subject else "This table"

    if scope_kind == "table":
        question_text = (
            f'{named} appears in {len(options)} different tables. '
            "Which one should I use?"
        )
    elif scope_kind == "subject":
        question_text = (
            f'{named} matches {len(options)} different '
            f'{label} values. Which one do you mean?'
        )
    elif scope_kind == "rows":
        if axis["column"].lower() == "is_isolated_cell":
            question_text = (
                "This table contains stray helper cells sitting outside the main "
                "table. Should they count towards the answer?"
            )
        else:
            question_text = (
                "This table mixes detail rows with pre-aggregated subtotal/total "
                "rows. Which rows should the answer be based on?"
            )
    else:
        question_text = (
            f'{named} is reported under {len(options)} different '
            f'{label} values. Which {label} should I use?'
        )

    note = ""
    if ambiguity["remaining_axes"]:
        note = (
            "After this, I will still need to narrow down: "
            + ", ".join(ambiguity["remaining_axes"])
            + "."
        )

    return {
        'question': user_question,
        'success': False,
        'needs_clarification': True,
        'clarification_question': question_text,
        'clarification_reason': (
            f'Read from {ambiguity["table"]}.{axis["column"]} — '
            + ("mixing these rows would double-count or pull in stray values."
               if scope_kind == "rows"
               else f'aggregating across different {label} values would mix '
                    f'incompatible figures.')
        ),
        'clarification_options': list(options),
        'option_labels': [label] * len(options),
        'secondary_note': note,
        'sql_query': None,
        'results': None,
        'warnings': [],
        'answer_summary': '',
        'trace': [
            (
                f'Scope check: "{ambiguity["subject_token"]}" matched '
                f'{len(ambiguity["subject_values"])} label value(s) in '
                f'{ambiguity["table"]}.{ambiguity["subject_column"]}'
                if ambiguity["subject_token"]
                else f'Scope check: no named figure; scanning {ambiguity["table"]}'
            ),
            f'Ambiguous {label}: {", ".join(str(v) for v in options)}',
        ],
        'error': None,
        'repair_attempts': 0,
        'interpreted_intent': {'route': 'scope_clarification'},
        'relevant_tables': [ambiguity["table"]],
        'query_plan': {},
    }


def build_schema_context(files_db: FilesDatabaseManager, user_question: str) -> str:
    """Route and render schema context for one free-text query."""
    if files_db is None:
        return ""
    route = route_schema_for_question(files_db, user_question)
    if not route.get("selected"):
        return ""
    return format_routed_schema_context(route)


def attach_schema_context_to_agent(
        sql_agent: FilesSQLAgent,
        files_db: FilesDatabaseManager,
) -> None:
    """Attach router metadata to the agent instance for introspection/debugging."""
    sql_agent.schema_router_index = list(
        getattr(files_db, "schema_router_index", []) or []
    )


class SQLAgentOrchestrator:
    """Facade used by the Streamlit app to coordinate DB, schema, and agent state."""

    def __init__(self, azure_client, deployment_name: str):
        self.azure_client = azure_client
        self.config = AgentOrchestratorConfig(deployment_name=deployment_name)
        self.query_memory: list[dict] = []
        self.verified_examples: list[dict] = []
        self._compiled_graph = None
        self._graph_compile_error = ""

    @staticmethod
    def langgraph_status() -> tuple[bool, str]:
        try:
            import langgraph  # noqa: F401

            return True, "available"
        except Exception as exc:
            return False, str(exc)

    def clear_memory(self) -> None:
        self.query_memory = []
        self.verified_examples = []

    def set_memory(self, memory_items: list[dict] | None) -> None:
        self.query_memory = list(memory_items or [])[-self.config.memory_limit:]

    def _remember_result(self, result: dict) -> list[dict]:
        item = compact_result_memory(result)
        if item:
            self.query_memory.append(item)
            self.query_memory = self.query_memory[-self.config.memory_limit:]
        example = compact_verified_example(result)
        if example:
            key = (example.get("question"), example.get("sql_query"))
            existing_keys = {
                (item.get("question"), item.get("sql_query"))
                for item in self.verified_examples
            }
            if key not in existing_keys:
                self.verified_examples.append(example)
                self.verified_examples = self.verified_examples[
                    -self.config.verified_examples_limit:
                ]
        return list(self.query_memory)

    def _node_load_memory(self, state: dict) -> dict:
        if not bool(state.get("enable_memory", True)):
            state.update(
                {
                    "memory_used": [],
                    "memory_context": "",
                    "memory_context_chars": 0,
                    "memory_context_est_tokens": 0,
                }
            )
            return _append_trace(state, "memory: disabled for this query")

        memory_items = list(state.get("graph_memory") or self.query_memory)
        memory_items = memory_items[-self.config.memory_limit:]
        prompt_memory = memory_items[-self.config.prompt_memory_limit:]
        memory_context = format_memory_context(
            prompt_memory,
            limit=self.config.prompt_memory_limit,
        )
        state.update(
            {
                "graph_memory": memory_items,
                "memory_used": prompt_memory,
                "memory_context": memory_context,
                "memory_context_chars": len(memory_context),
                "memory_context_est_tokens": estimate_prompt_tokens(memory_context),
            }
        )
        return _append_trace(
            state,
            f"memory: loaded {len(prompt_memory)} bounded prior query record(s)",
        )

    def _node_route_schema(self, state: dict) -> dict:
        files_db = state["files_db"]
        question = str(state.get("question") or "")
        route = route_schema_for_question(
            files_db,
            question,
            max_tables=self.config.max_routed_tables,
            max_columns=self.config.max_routed_columns,
        )
        schema_expanded = False
        if (
                route.get("confidence") in {"low", "medium"}
                or route.get("ambiguous")
        ) and int(route.get("available_table_count") or 0) > len(route.get("selected") or []):
            expanded_route = route_schema_for_question(
                files_db,
                question,
                max_tables=self.config.expanded_routed_tables,
                max_columns=self.config.expanded_routed_columns,
            )
            if len(expanded_route.get("selected") or []) > len(route.get("selected") or []):
                route = expanded_route
                schema_expanded = True

        selected = route.get("selected") or []
        schema_context = format_routed_schema_context(route) if selected else ""
        memory_context = str(state.get("memory_context") or "")
        selected_tables = [
            str(item.get("table", {}).get("table_name", ""))
            for item in selected
        ]
        selected_tables = [name for name in selected_tables if name]
        verified_examples = select_verified_examples(
            self.verified_examples,
            question,
            selected_tables,
            limit=self.config.prompt_example_limit,
        )
        examples_context = format_verified_examples_context(verified_examples)
        prompt_context = schema_context
        if memory_context:
            prompt_context = (
                f"{schema_context}\n\n{memory_context}"
                if schema_context
                else memory_context
            )
        if examples_context:
            prompt_context = (
                f"{prompt_context}\n\n{examples_context}"
                if prompt_context
                else examples_context
            )
        state.update(
            {
                "schema_route": route,
                "schema_context": schema_context,
                "prompt_context": prompt_context,
                "selected_tables": selected_tables,
                "schema_confidence": route.get("confidence", "unknown"),
                "schema_expanded": schema_expanded,
                "schema_top_score": route.get("top_score", 0),
                "schema_score_gap": route.get("score_gap", 0),
                "verified_examples_used": verified_examples,
                "verified_examples_context_chars": len(examples_context),
                "verified_examples_context_est_tokens": estimate_prompt_tokens(examples_context),
                "schema_context_chars": len(schema_context),
                "schema_context_est_tokens": estimate_prompt_tokens(schema_context),
                "prompt_context_chars": len(prompt_context),
                "prompt_context_est_tokens": estimate_prompt_tokens(prompt_context),
            }
        )
        return _append_trace(
            state,
            "schema: routed "
            f"{len(selected_tables)} table(s), "
            f"confidence={route.get('confidence', 'unknown')}, "
            f"expanded={schema_expanded}, "
            f"examples={len(verified_examples)}, "
            f"~{estimate_prompt_tokens(prompt_context)} prompt token(s)",
        )

    def _should_skip_clarification(self, state: dict) -> bool:
        """Skip clarification only when the router is confident AND unambiguous.

        This gates *table-selection* ambiguity, which is all the router can
        measure; value/metric ambiguity is not covered, so the gate is
        deliberately conservative (defaults to the top confidence band only).
        """
        if not bool(getattr(self.config, "gate_clarification_on_confidence", False)):
            return False
        route = state.get("schema_route") or {}
        if route.get("ambiguous"):
            return False
        confidence = str(
            state.get("schema_confidence") or route.get("confidence") or "unknown"
        )
        threshold = str(getattr(self.config, "clarification_skip_confidence", "high"))
        rank = {"low": 0, "medium": 1, "high": 2}
        # Require a real selection to skip on.
        if not (route.get("selected") or state.get("selected_tables")):
            return False
        return rank.get(confidence, -1) >= rank.get(threshold, 2)

    def _node_scope_check(self, state: dict) -> dict:
        """Ask about mixed unit/currency/segment scope before any SQL runs.

        This is deliberately independent of the router's confidence gate: the
        router only measures *table* ambiguity, while this catches a figure
        that exists many times inside one table under different scopes.
        """
        if not bool(getattr(self.config, "detect_scope_ambiguity", True)):
            return state

        question = str(state.get("question") or "")
        try:
            ambiguity = detect_scope_ambiguity(
                state["files_db"], question, state.get("schema_route")
            )
        except Exception as exc:
            return _append_trace(state, f"scope: check skipped ({exc})")

        if not ambiguity:
            return _append_trace(state, "scope: no ambiguous axis found")

        def extend_prompt(block: str) -> None:
            prompt_context = str(state.get("prompt_context") or "")
            state["prompt_context"] = (
                f"{prompt_context}\n\n{block}" if prompt_context else block
            )
            state["prompt_context_chars"] = len(state["prompt_context"])
            state["prompt_context_est_tokens"] = estimate_prompt_tokens(state["prompt_context"])

        # Choices the user already made become hard SQL requirements, so the
        # generator cannot quietly widen the scope back out again.
        resolved = ambiguity.get("resolved_filters") or []
        if resolved:
            extend_prompt(
                "MANDATORY FILTERS (chosen by the user — the SQL MUST include all):\n"
                + "\n".join(f"- {clause}" for clause in resolved)
            )
            state["scope_filters"] = resolved

        # ── Grouping mode: never block, split the result instead ────────────
        # Carrying the scope columns through SELECT/GROUP BY keeps every row
        # tagged with its own unit, so an aggregate can never mix incompatible
        # figures and the answer can read the unit off a column.
        open_axes = (ambiguity.get("open_axes") or [])[
            : max(1, int(getattr(self.config, "max_scope_group_columns", 4)))
        ]
        if str(getattr(self.config, "scope_mode", "group")) == "group" and open_axes:
            columns = [axis["column"] for axis in open_axes]
            detail = "; ".join(
                f'"{axis["column"]}" ({", ".join(str(v) for v in axis["values"][:4])})'
                for axis in open_axes
            )
            extend_prompt(
                "MANDATORY GROUPING (these columns distinguish incompatible figures):\n"
                + "\n".join(f'- "{column}"' for column in columns)
                + "\nThe SQL MUST list every one of them in SELECT and in GROUP BY, "
                  "and MUST NOT aggregate across them — one output row per "
                  "combination. Values differ per group: "
                + detail
                + "."
            )
            state["scope_group_by"] = columns
            return _append_trace(
                state,
                f"scope: splitting result by {', '.join(columns)} "
                f"instead of asking ({len(resolved)} filter(s) already pinned)",
            )

        skipped = ambiguity.get("skipped_axes") or []
        if skipped:
            state["scope_skipped_axes"] = skipped
            return _append_trace(
                state,
                "scope: user chose to answer anyway; "
                f"{len(skipped)} axis/axes left open "
                f"({', '.join(a['label'] for a in skipped)})",
            )

        if not ambiguity.get("axis"):
            return _append_trace(
                state,
                f"scope: fully pinned ({len(resolved)} filter(s) applied)"
                if resolved else "scope: no ambiguous axis found",
            )

        state["result"] = scope_clarification_result(question, ambiguity)
        state["scope_ambiguity"] = ambiguity
        return _append_trace(
            state,
            f'scope: asking about {ambiguity["axis"]["label"]} '
            f'({len(ambiguity["options"])} option(s))'
            + (f' for "{ambiguity["subject_token"]}"' if ambiguity["subject_token"] else ""),
        )

    def _node_run_agent(self, state: dict) -> dict:
        # The scope check may already have produced a clarification; if so the
        # question is not answerable yet and no SQL should be generated.
        if state.get("result"):
            return _append_trace(state, "agent: skipped (scope clarification pending)")

        sql_agent = state["sql_agent"]
        question = str(state.get("question") or "")
        prompt_context = str(state.get("prompt_context") or "")
        skip_clarification = self._should_skip_clarification(state)
        state["clarification_skipped"] = skip_clarification
        result = sql_agent.execute_query_with_explanation(
            question,
            schema_context=prompt_context or None,
            skip_clarification=skip_clarification,
        )
        # Answering without narrowing is allowed, but never silent: say which
        # axes were combined so a mixed figure cannot pass as a clean one.
        for axis in state.get("scope_skipped_axes") or []:
            values = ", ".join(str(v) for v in axis["values"][:5])
            result.setdefault("warnings", []).append(
                f"Scope not narrowed: this answer combines all {axis['label']} "
                f"values ({values}). Figures from different {axis['label']} "
                f"values are not directly comparable."
            )

        state["result"] = result
        if skip_clarification:
            state = _append_trace(
                state,
                "clarification: skipped (confidence="
                f"{state.get('schema_confidence', 'unknown')}, unambiguous route)",
            )
        return _append_trace(
            state,
            "agent: "
            + (
                "clarification requested"
                if result.get("needs_clarification")
                else "query succeeded"
                if result.get("success")
                else "query failed"
            ),
        )

    def _node_store_memory(self, state: dict) -> dict:
        if not bool(state.get("enable_memory", True)):
            state["graph_memory"] = list(self.query_memory)
            return _append_trace(state, "memory: not updated")

        result = dict(state.get("result") or {})
        memory_items = self._remember_result(result)
        state["graph_memory"] = memory_items
        state["result"] = result
        return _append_trace(
            state,
            f"memory: stored compact result ({len(memory_items)} total record(s))",
        )

    def _node_result_sanity(self, state: dict) -> dict:
        result = dict(state.get("result") or {})
        sanity = analyze_result_sanity(result)
        result["result_sanity"] = sanity
        state["result"] = result
        status = sanity.get("status", "unknown")
        flags = sanity.get("flags") or []
        detail = f"; flags={', '.join(flags)}" if flags else ""
        return _append_trace(state, f"sanity: {status}{detail}")

    def _local_graph_invoke(self, state: dict) -> dict:
        for node in (
                self._node_load_memory,
                self._node_route_schema,
                self._node_scope_check,
                self._node_run_agent,
                self._node_result_sanity,
                self._node_store_memory,
        ):
            state = node(state)
        return state

    def _compile_langgraph(self):
        if self._compiled_graph is not None:
            return self._compiled_graph
        if self._graph_compile_error:
            return None

        try:
            from langgraph.graph import END, START, StateGraph

            try:
                from langgraph.checkpoint.memory import InMemorySaver
            except ImportError:
                from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

            workflow = StateGraph(dict)
            workflow.add_node("load_memory", self._node_load_memory)
            workflow.add_node("route_schema", self._node_route_schema)
            workflow.add_node("scope_check", self._node_scope_check)
            workflow.add_node("run_agent", self._node_run_agent)
            workflow.add_node("result_sanity", self._node_result_sanity)
            workflow.add_node("store_memory", self._node_store_memory)
            workflow.add_edge(START, "load_memory")
            workflow.add_edge("load_memory", "route_schema")
            workflow.add_edge("route_schema", "scope_check")
            workflow.add_edge("scope_check", "run_agent")
            workflow.add_edge("run_agent", "result_sanity")
            workflow.add_edge("result_sanity", "store_memory")
            workflow.add_edge("store_memory", END)
            self._compiled_graph = workflow.compile(checkpointer=InMemorySaver())
        except Exception as exc:
            self._graph_compile_error = str(exc)
            self._compiled_graph = None
        return self._compiled_graph

    def _invoke_query_graph(
            self,
            state: dict,
            use_langgraph: bool,
    ) -> dict:
        if use_langgraph:
            graph = self._compile_langgraph()
            if graph is not None:
                try:
                    thread_id = str(state.get("thread_id") or "sql-agent-session")
                    final_state = graph.invoke(
                        state,
                        config={"configurable": {"thread_id": thread_id}},
                    )
                    final_state["graph_mode"] = "langgraph"
                    return final_state
                except Exception as exc:
                    state = _append_trace(
                        state,
                        f"langgraph: runtime fallback ({exc})",
                    )

        state = self._local_graph_invoke(state)
        state["graph_mode"] = (
            "local_fallback"
            if use_langgraph
            else "local_graph"
        )
        if use_langgraph and self._graph_compile_error:
            state = _append_trace(
                state,
                f"langgraph: unavailable ({self._graph_compile_error})",
            )
        elif use_langgraph:
            state = _append_trace(state, "langgraph: unavailable")
        return state

    def build_agent(
            self,
            files_db: FilesDatabaseManager,
            previous_agent: FilesSQLAgent | None = None,
    ) -> FilesSQLAgent:
        agent = FilesSQLAgent(
            self.azure_client,
            files_db,
            self.config.deployment_name,
        )
        attach_schema_context_to_agent(agent, files_db)
        if previous_agent is not None:
            agent.conversation_history = list(
                getattr(previous_agent, "conversation_history", []) or []
            )
            agent.last_query_result = getattr(previous_agent, "last_query_result", None)
            agent.last_query_context = getattr(previous_agent, "last_query_context", None)
        return agent

    def refresh_agent_schema(
            self,
            sql_agent: FilesSQLAgent | None,
            files_db: FilesDatabaseManager,
    ) -> None:
        if sql_agent is None:
            return
        sql_agent.files_db = files_db
        if hasattr(sql_agent, "refresh_schema_info"):
            sql_agent.refresh_schema_info()
        else:
            sql_agent.schema_info = files_db.get_schema_info()
        attach_schema_context_to_agent(sql_agent, files_db)

    def refresh_schema_artifacts(
            self,
            files_db: FilesDatabaseManager,
            sql_agent: FilesSQLAgent | None = None,
    ) -> None:
        refresh_flat_file_schema_artifacts(files_db)
        self.refresh_agent_schema(sql_agent, files_db)

    def replace_table(
            self,
            files_db: FilesDatabaseManager,
            table_name: str,
            df: pd.DataFrame,
            source_file_name: str,
            source_sheet_name: str,
            sql_agent: FilesSQLAgent | None = None,
    ) -> str:
        replaced_table = replace_table_in_files_db(
            files_db,
            table_name,
            df,
            source_file_name,
            source_sheet_name,
        )
        if replaced_table:
            self.refresh_schema_artifacts(files_db, sql_agent=sql_agent)
        return replaced_table

    def run_free_text_query(
            self,
            sql_agent: FilesSQLAgent,
            files_db: FilesDatabaseManager,
            user_question: str,
            graph_memory: list[dict] | None = None,
            use_langgraph: bool = True,
            enable_memory: bool = True,
            thread_id: str = "sql-agent-session",
    ) -> dict:
        if enable_memory:
            self.set_memory(graph_memory if graph_memory is not None else self.query_memory)
        state = self._invoke_query_graph(
            {
                "question": user_question,
                "sql_agent": sql_agent,
                "files_db": files_db,
                "graph_memory": list(self.query_memory),
                "thread_id": thread_id,
                "enable_memory": enable_memory,
                "graph_trace": [],
            },
            use_langgraph=use_langgraph,
        )
        result = dict(state.get("result") or {})
        for key in (
                "schema_context_chars", "schema_context_est_tokens",
                "memory_context_chars", "memory_context_est_tokens",
                "verified_examples_context_chars", "verified_examples_context_est_tokens",
                "prompt_context_chars", "prompt_context_est_tokens",
        ):
            result[key] = int(state.get(key) or 0)
        result["schema_context_applied"] = bool(state.get("schema_context"))
        result["graph_mode"] = state.get("graph_mode", "local_graph")
        result["graph_trace"] = list(state.get("graph_trace") or [])
        result["graph_memory"] = list(state.get("graph_memory") or self.query_memory)
        result["memory_used"] = list(state.get("memory_used") or [])
        result["selected_schema_tables"] = list(state.get("selected_tables") or [])
        result["schema_confidence"] = state.get("schema_confidence", "unknown")
        result["schema_expanded"] = bool(state.get("schema_expanded"))
        result["clarification_skipped"] = bool(state.get("clarification_skipped"))
        result["schema_top_score"] = state.get("schema_top_score", 0)
        result["schema_score_gap"] = state.get("schema_score_gap", 0)
        result["verified_examples_used"] = list(state.get("verified_examples_used") or [])
        return result

    def run_guided_query(
            self,
            sql_agent: FilesSQLAgent,
            guided_payload: dict,
            enable_memory: bool = True,
    ) -> dict:
        result = sql_agent.execute_guided_query(guided_payload)
        result["graph_mode"] = "guided_direct"
        result["graph_memory"] = (
            self._remember_result(result)
            if enable_memory
            else list(self.query_memory)
        )
        result["result_sanity"] = analyze_result_sanity(result)
        result["graph_trace"] = [
            "guided: deterministic route",
            f"sanity: {result['result_sanity'].get('status', 'unknown')}",
            "memory: stored compact result" if enable_memory else "memory: not updated",
        ]
        result.setdefault("prompt_context_est_tokens", 0)
        result.setdefault("schema_context_est_tokens", 0)
        result.setdefault("memory_context_est_tokens", 0)
        result.setdefault("verified_examples_context_est_tokens", 0)
        return result
