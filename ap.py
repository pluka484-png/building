import streamlit as st
import os
import html
from pathlib import Path
import io
from datetime import datetime
from dotenv import load_dotenv
import hashlib
from typing import Any

# ── Streamlit version compatibility ─────────────────────────────────────────
# The app must also run on older Streamlit releases (e.g. 1.23 installed via
# Homebrew). Backfill APIs that newer code paths use unconditionally:
#   - st.rerun was added in 1.27 (previously st.experimental_rerun)
#   - st.toggle was added in 1.26 (st.checkbox is the functional equivalent)
if not hasattr(st, "rerun") and hasattr(st, "experimental_rerun"):
    st.rerun = st.experimental_rerun
if not hasattr(st, "toggle"):
    st.toggle = st.checkbox

# Load .env from the same directory as THIS file — must happen before any
# azure.identity import so env vars are in the process environment in time.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH, override=True)

from sql_agent_core import (
    FilesDatabaseManager,
    SQLAgentOrchestrator,
    initialize_azure_client,
    load_dataframe_into_files_db,
    refresh_flat_file_schema_artifacts,
    route_schema_for_question as core_route_schema_for_question,
)
import pandas as pd
from flat_file_builder import (
    drop_empty_or_zero_columns,
    extract_and_flatten_sheet,
    extract_workbook_sheets,
    finalize_extracted_sheet,
    apply_pandas_cleanup,
    clean_column_name,
    to_excel_bytes,
    _validate_workbook_archive,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Azure OpenAI Configuration — read from env if set, otherwise keep hardcoded defaults.
GPT_ENDPOINT   = os.getenv("GPT_ENDPOINT",   "https://cog-bnl-0001-prp-ext002-oai.openai.azure.com/")
GPT_DEPLOYMENT = os.getenv("GPT_DEPLOYMENT", "gpt-5.1-Finance")
GPT_API_VERSION = os.getenv("GPT_API_VERSION", "2024-12-01-preview")


# ── Config debug logging (True/False only — no secrets printed) ────────────
print(f"[app] .env path        : {_ENV_PATH}")
print(f"[app] .env exists       : {_ENV_PATH.exists()}")
print(f"[app] GPT_ENDPOINT set  : {bool(GPT_ENDPOINT)}")
print(f"[app] GPT_DEPLOYMENT set: {bool(GPT_DEPLOYMENT)}")
print(f"[app] GPT_API_VERSION   : {bool(GPT_API_VERSION)}")


# Custom CSS for better styling
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
   
    /* Header styling */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
   
    h2 {
        color: #2563eb;
        font-weight: 600;
    }
   
    h3 {
        color: #3b82f6;
    }
   
    /* Card-like containers */
    .stExpander {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
   
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
   
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
   
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
   
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
        font-size: 1rem;
    }
   
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
   
    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
        padding: 1rem;
    }
   
    /* Code block styling */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
   
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
   
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
   
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }
   
    /* Metrics container */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
   
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
   
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }
</style>
"""


# ============================================================================
# BENELUX-AWARE EXTRACTION HELPER
# ============================================================================

def _extract_sheet_safely(
    file_bytes: bytes,
    sheet_name: str,
    preferred_profile: str = "auto",
) -> pd.DataFrame:
    """Extract one sheet, falling back to a direct flat read on failure."""
    try:
        df = extract_and_flatten_sheet(file_bytes, sheet_name, preferred_profile=preferred_profile)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"[extract_sheet] Structured extraction failed: {e}")

    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
        if df is not None and not df.empty:
            return finalize_extracted_sheet(df, strip_text=True, split_hierarchy=True)
    except Exception as e:
        print(f"[extract_sheet] Flat fallback failed: {e}")

    return pd.DataFrame()


# ============================================================================
# SESSION STATE
# ============================================================================


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.azure_client = None
        st.session_state.azure_config = None
        st.session_state.agent_orchestrator = None
        st.session_state.files_db = None
        st.session_state.sql_agent = None
        st.session_state.query_history = []
        st.session_state.graph_memory = []
        st.session_state.use_graph_orchestration = True
        st.session_state.graph_thread_id = hashlib.sha256(
            str(datetime.now()).encode()
        ).hexdigest()[:12]
        st.session_state.uploaded_files = []
        st.session_state.files_loaded = False
        st.session_state.file_sheets = {}  # Maps file names to their sheets
        st.session_state.selected_sheets = {}  # Maps file names to selected sheets
        st.session_state.flat_file_overrides = {}  # Maps Excel file names to 'already flat' boolean
        st.session_state.show_sheet_selector = False

    # Backfill keys for existing sessions after app updates.
    if 'flat_file_overrides' not in st.session_state:
        st.session_state.flat_file_overrides = {}
    if 'agent_orchestrator' not in st.session_state:
        st.session_state.agent_orchestrator = None
    if 'graph_memory' not in st.session_state:
        st.session_state.graph_memory = []
    if 'use_graph_orchestration' not in st.session_state:
        st.session_state.use_graph_orchestration = True
    if 'graph_thread_id' not in st.session_state:
        st.session_state.graph_thread_id = hashlib.sha256(
            str(datetime.now()).encode()
        ).hexdigest()[:12]
    if 'pending_clarified_question' not in st.session_state:
        st.session_state.pending_clarified_question = ""


# ============================================================================
# INITIALIZATION
# ============================================================================


def get_excel_sheets_from_bytes(file_bytes: bytes):
    """Extract sheet names directly from Excel bytes (no temp file needed)."""
    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
        sheets = excel_file.sheet_names
        excel_file.close()
        return sheets
    except Exception:
        return []


def _read_excel_sheets_from_bytes(file_bytes: bytes, sheet_names: list[str]) -> dict[str, pd.DataFrame]:
    """Read multiple sheets in one workbook parse to reduce repeated IO overhead."""
    if not sheet_names:
        return {}

    try:
        with pd.ExcelFile(io.BytesIO(file_bytes)) as excel_file:
            result = pd.read_excel(excel_file, sheet_name=sheet_names)
    except Exception:
        # Keep behavior resilient: if batch read fails, fall back per sheet.
        result = {}
        for sheet_name in sheet_names:
            try:
                result[sheet_name] = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
            except Exception:
                result[sheet_name] = pd.DataFrame()

    if isinstance(result, pd.DataFrame):
        return {sheet_names[0]: pd.DataFrame(result)}

    return {str(name): pd.DataFrame(df) for name, df in result.items()}


def _uploaded_file_name(uploaded_file: Any) -> str:
    name = getattr(uploaded_file, "name", None)
    return str(name) if isinstance(name, str) and name else "uploaded_file"


def _uploaded_file_bytes(uploaded_file: Any) -> bytes:
    if hasattr(uploaded_file, "getbuffer"):
        try:
            return bytes(uploaded_file.getbuffer())
        except Exception:
            pass
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
    if hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
    raise ValueError("Unsupported uploaded file object: expected Streamlit UploadedFile-like input")


def analyze_uploaded_files(uploaded_files: list[Any] | None):
    """Analyze uploaded files and extract sheet information"""
    if not uploaded_files:
        return {}

    file_sheets = {}
   
    for uploaded_file in uploaded_files:
        file_name = _uploaded_file_name(uploaded_file)
        file_extension = os.path.splitext(file_name)[1].lower()

        if file_extension in ['.xlsx', '.xls']:
            file_bytes = _uploaded_file_bytes(uploaded_file)
            try:
                _validate_workbook_archive(file_bytes)
            except ValueError as exc:
                st.error(f"'{file_name}' was rejected: {exc}")
                continue
            except Exception:
                pass  # non-zip (e.g. legacy .xls); let pandas decide below
            sheets = get_excel_sheets_from_bytes(file_bytes)
            if sheets:
                file_sheets[file_name] = {
                    'sheets': sheets,
                    'path': None,
                    'type': 'excel'
                }
        elif file_extension == '.csv':
            file_sheets[file_name] = {
                'sheets': None,
                'path': None,
                'type': 'csv'
            }

    return file_sheets


def _apply_default_pandas_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic pandas cleanup defaults used by app loading paths."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    return pd.DataFrame(
        apply_pandas_cleanup(
            df,
            drop_columns=[],
            rename_map={},
            split_config={},
            drop_blank_columns=[],
            type_conversions={},
            strip_text=True,
            clean_names=False,
            dash_split_mode="spaced",
        )
    )


def _apply_flat_file_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight cleanup for user-confirmed flat files.

    Keeps processing minimal and removes columns containing only blank/zero
    values before SQLite and schema generation see them.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    cleaned = drop_empty_or_zero_columns(pd.DataFrame(df).copy())
    text_cols = [
        col for col in cleaned.columns
        if pd.api.types.is_object_dtype(cleaned[col].dtype) or pd.api.types.is_string_dtype(cleaned[col].dtype)
    ]
    if text_cols:
        cleaned[text_cols] = cleaned[text_cols].apply(lambda s: s.astype(str).str.strip())

    return pd.DataFrame(cleaned)


def _display_schema_route_preview(files_db: FilesDatabaseManager, user_question: str) -> None:
    """Show the local table/column route that will be used for a free-text query."""
    if not user_question.strip():
        return
    route = core_route_schema_for_question(files_db, user_question)
    selected = route.get("selected", [])
    if not selected:
        return

    with st.expander("🧭 Schema route preview", expanded=False):
        st.caption(
            f"{len(selected)} table(s) selected from "
            f"{route.get('available_table_count', 0)} available"
            + (" · ambiguous match" if route.get("ambiguous") else "")
        )
        for item in selected:
            table = item.get("table", {})
            table_name = str(table.get("table_name", ""))
            st.markdown(
                f"**{table_name}** · {int(table.get('row_count', 0) or 0):,} rows"
            )
            columns = []
            for col in item.get("columns", []):
                samples = col.get("samples") or []
                columns.append({
                    "column": col.get("name", ""),
                    "dtype": col.get("dtype", ""),
                    "role": col.get("role", "dimension"),
                    "examples": ", ".join(str(v) for v in samples[:3]),
                })
            if columns:
                st.dataframe(pd.DataFrame(columns), use_container_width=True, hide_index=True)


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _warning_is_financial_guardrail(warning: str) -> bool:
    text = str(warning or "").lower()
    return any(
        phrase in text
        for phrase in (
            "multiple unit",
            "multiple currency",
            "multiple value_kind",
            "spans multiple",
            "subtotal/total",
            "pre-aggregated totals",
        )
    )


def _refined_question_from_clarification(result: dict, option: str, label: str = "") -> str:
    base_question = str(result.get("question") or "").strip()
    clarification = str(result.get("clarification_question") or "").strip()
    chosen = f"{label}: {option}" if label else option
    parts = [base_question]
    if clarification:
        parts.append(f"Clarification requested: {clarification}")
    parts.append(f"Use this interpretation: {chosen}")
    return "\n\n".join(part for part in parts if part)


def load_data(
    uploaded_files: list[Any] | None,
    selected_sheets=None,
    existing_files_db: FilesDatabaseManager | None = None,
    flat_file_overrides: dict[str, bool] | None = None,
):
    """Load uploaded files into the database"""
    if not uploaded_files:
        st.error("❌ No files were provided for loading.")
        return None

    try:
        with st.spinner("Loading files..."):
            # Reuse existing in-memory DB so newly added files do not wipe previous ones.
            files_db = existing_files_db if existing_files_db is not None else FilesDatabaseManager()
            loaded_count = 0
           
            for uploaded_file in uploaded_files:
                file_name = _uploaded_file_name(uploaded_file)
                file_bytes = _uploaded_file_bytes(uploaded_file)
                file_extension = os.path.splitext(file_name)[1].lower()
                force_flat = bool(flat_file_overrides and flat_file_overrides.get(file_name, False))

                if file_extension in ['.xlsx', '.xls']:
                    try:
                        _validate_workbook_archive(file_bytes)
                    except ValueError as exc:
                        st.error(f"❌ '{file_name}' was rejected: {exc}")
                        continue
                    except Exception:
                        pass  # non-zip (e.g. legacy .xls); let pandas decide below

                # Explicit user override: treat selected Excel files as already flat.
                if file_extension in ['.xlsx', '.xls'] and force_flat:
                    if selected_sheets and file_name in selected_sheets:
                        sheets_to_process = selected_sheets[file_name]
                    else:
                        sheets_to_process = get_excel_sheets_from_bytes(file_bytes)

                    if not sheets_to_process:
                        st.warning(f"⚠️ No sheets selected/found for: {file_name}")
                        continue

                    preloaded_flat_sheets = _read_excel_sheets_from_bytes(file_bytes, sheets_to_process)

                    loaded_sheet_mappings = []
                    for sheet_name in sheets_to_process:
                        try:
                            flat_df = preloaded_flat_sheets.get(sheet_name, pd.DataFrame())
                            if flat_df is None or flat_df.empty:
                                flat_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
                            if flat_df is None or flat_df.empty:
                                continue
                            flat_df = _apply_flat_file_cleanup(pd.DataFrame(flat_df))
                            table_name = load_dataframe_into_files_db(
                                files_db,
                                flat_df,
                                file_name,
                                sheet_name,
                            )
                            if table_name:
                                loaded_sheet_mappings.append(f"{sheet_name} -> {table_name}")
                        except Exception as e:
                            st.warning(f"⚠️ Failed to load flat sheet '{sheet_name}' from '{file_name}': {e}")

                    if loaded_sheet_mappings:
                        files_db.loaded_files.append({
                            'file_path': file_name,
                            'type': 'Excel',
                            'sheets': loaded_sheet_mappings,
                        })
                        loaded_count += 1
                        st.success(f"✅ Loaded flat file: {file_name} ({len(loaded_sheet_mappings)} sheet(s))")
                    else:
                        st.warning(f"⚠️ Failed to load: {file_name}")
                    continue

                # Fast path: process selected Excel sheets directly in-memory and load into DB.
                # This avoids flatten->write temp workbook->read workbook roundtrip.
                if file_extension in ['.xlsx', '.xls'] and not force_flat:
                    if selected_sheets and file_name in selected_sheets:
                        sheets_to_process = selected_sheets[file_name]
                    else:
                        sheets_to_process = get_excel_sheets_from_bytes(file_bytes)

                    if not sheets_to_process:
                        st.warning(f"⚠️ No sheets selected/found for: {file_name}")
                        continue

                    try:
                        extracted_sheets = extract_workbook_sheets(
                            file_bytes,
                            sheets_to_process,
                            preferred_profile="auto",
                        )
                    except Exception as exc:
                        st.warning(
                            "⚠️ Batch extraction failed; retrying selected sheets "
                            f"individually. ({exc})"
                        )
                        extracted_sheets = {}

                    prepared_sheets: dict[str, pd.DataFrame] = {}
                    failed_sheet_names: list[str] = []
                    fallback_sheets: dict[str, pd.DataFrame] | None = None

                    for sheet_name in sheets_to_process:
                        st.write(f"🔄 {sheet_name}: preparing flat data")
                        structured_df = extracted_sheets.get(
                            sheet_name,
                            pd.DataFrame(),
                        )
                        if structured_df is None or structured_df.empty:
                            structured_df = _extract_sheet_safely(
                                file_bytes,
                                sheet_name,
                                preferred_profile="auto",
                            )

                        if structured_df is None or structured_df.empty:
                            if fallback_sheets is None:
                                fallback_sheets = _read_excel_sheets_from_bytes(
                                    file_bytes,
                                    sheets_to_process,
                                )
                            fallback_df = fallback_sheets.get(sheet_name, pd.DataFrame())
                            if fallback_df is None or fallback_df.empty:
                                fallback_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
                            if fallback_df is None or fallback_df.empty:
                                failed_sheet_names.append(sheet_name)
                                st.warning(f"⚠️ Sheet '{sheet_name}' is empty, skipping...")
                                continue
                            structured_df = pd.DataFrame(fallback_df)
                            structured_df = finalize_extracted_sheet(structured_df, strip_text=True, split_hierarchy=True)

                        structured_df = drop_empty_or_zero_columns(
                            _apply_default_pandas_cleanup(
                                pd.DataFrame(structured_df)
                            )
                        )
                        if structured_df.empty:
                            failed_sheet_names.append(sheet_name)
                            st.warning(f"⚠️ Sheet '{sheet_name}' produced no rows after cleanup, skipping...")
                            continue

                        prepared_sheets[sheet_name] = structured_df

                    if prepared_sheets:
                        st.success(
                            f"✅ Prepared {len(prepared_sheets)} sheet(s) for DB load"
                            + (f" | skipped: {len(failed_sheet_names)}" if failed_sheet_names else "")
                        )
                    elif failed_sheet_names:
                        st.warning(f"⚠️ No sheets prepared from '{file_name}'. Skipped: {len(failed_sheet_names)}")

                    # Phase 2: load prepared sheets into DB.
                    loaded_sheet_mappings = []
                    for sheet_name, structured_df in prepared_sheets.items():
                        try:
                            table_name = load_dataframe_into_files_db(
                                files_db,
                                structured_df,
                                file_name,
                                sheet_name,
                            )
                            if table_name:
                                loaded_sheet_mappings.append(f"{sheet_name} -> {table_name}")
                        except Exception as e:
                            st.warning(f"⚠️ Failed to load prepared sheet '{sheet_name}' into DB: {e}")

                    if loaded_sheet_mappings:
                        files_db.loaded_files.append({
                            'file_path': file_name,
                            'type': 'Excel',
                            'sheets': loaded_sheet_mappings,
                        })
                        loaded_count += 1
                        st.success(f"✅ Loaded: {file_name} ({len(loaded_sheet_mappings)} sheet(s))")
                    else:
                        st.warning(f"⚠️ Failed to load: {file_name}")
                    continue

                if file_extension == '.csv':
                    try:
                        csv_df = pd.read_csv(io.BytesIO(file_bytes))
                        csv_df = _apply_flat_file_cleanup(csv_df)
                        table_name = load_dataframe_into_files_db(
                            files_db,
                            csv_df,
                            file_name,
                            Path(file_name).stem or "flat_table",
                        )
                    except Exception as exc:
                        table_name = ""
                        st.warning(f"⚠️ Failed to load CSV '{file_name}': {exc}")
                    if table_name:
                        files_db.loaded_files.append(
                            {
                                "file_path": file_name,
                                "type": "CSV",
                                "sheets": [f"{file_name} -> {table_name}"],
                            }
                        )
                        loaded_count += 1
                        st.success(f"✅ Loaded: {file_name}")
                    else:
                        st.warning(f"⚠️ Failed to load: {file_name}")
                    continue

                st.warning(f"⚠️ Unsupported file type: {file_name}")

            if loaded_count > 0:
                try:
                    refresh_flat_file_schema_artifacts(files_db)
                except Exception as exc:
                    st.warning(
                        "⚠️ Data loaded, but the AI schema context/common workbook "
                        f"could not be refreshed: {exc}"
                    )
                return files_db
            if existing_files_db is not None and files_db.tables_info:
                st.info("ℹ️ No new tables were added, keeping existing loaded files.")
                return files_db
            st.error("❌ No files were loaded successfully.")
            return None
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None


def initialize_azure():
    """Initialize Azure OpenAI client only."""
    if not st.session_state.initialized:
        with st.spinner("Initializing Azure OpenAI..."):
            # .env is already loaded at module level via __file__.
            # No second load_dotenv() call needed here.
            try:
                azure_client, azure_config = initialize_azure_client(
                    GPT_ENDPOINT,
                    GPT_DEPLOYMENT,
                    GPT_API_VERSION,
                )
                st.session_state.azure_client = azure_client
                st.session_state.azure_config = azure_config
                st.session_state.agent_orchestrator = SQLAgentOrchestrator(
                    azure_client,
                    azure_config.deployment_name,
                )
                st.session_state.agent_orchestrator.set_memory(
                    st.session_state.graph_memory
                )
                st.session_state.initialized = True
                return True
            except Exception as e:
                st.error(f"Failed to initialize Azure OpenAI client: {e}")
                return False

    if st.session_state.agent_orchestrator is None and st.session_state.azure_client is not None:
        st.session_state.agent_orchestrator = SQLAgentOrchestrator(
            st.session_state.azure_client,
            st.session_state.azure_config.deployment_name,
        )
        st.session_state.agent_orchestrator.set_memory(
            st.session_state.graph_memory
        )
    return True


def load_uploaded_files(uploaded_files, selected_sheets=None, flat_file_overrides=None):
    """Load uploaded files and initialize SQL agent"""
    if uploaded_files:
        existing_db = st.session_state.files_db if st.session_state.files_loaded else None
        files_db = load_data(
            uploaded_files,
            selected_sheets,
            existing_files_db=existing_db,
            flat_file_overrides=flat_file_overrides,
        )

        if files_db is None or not files_db.tables_info:
            st.error("No data was loaded. Please check your files.")
            return False

        orchestrator = st.session_state.agent_orchestrator
        if orchestrator is None:
            orchestrator = SQLAgentOrchestrator(
                st.session_state.azure_client,
                st.session_state.azure_config.deployment_name,
            )
            orchestrator.set_memory(st.session_state.graph_memory)
            st.session_state.agent_orchestrator = orchestrator

        sql_agent = orchestrator.build_agent(
            files_db,
            previous_agent=st.session_state.sql_agent,
        )

        st.session_state.files_db = files_db
        st.session_state.sql_agent = sql_agent

        # Keep previously uploaded files listed and add any new ones.
        previous_uploaded = st.session_state.uploaded_files or []
        merged_uploaded = list(previous_uploaded)
        seen_uploaded_names = {
            _uploaded_file_name(item)
            for item in merged_uploaded
        }
        for item in uploaded_files:
            item_name = _uploaded_file_name(item)
            if item_name not in seen_uploaded_names:
                merged_uploaded.append(item)
                seen_uploaded_names.add(item_name)
        st.session_state.uploaded_files = merged_uploaded

        st.session_state.files_loaded = True
        st.session_state.show_sheet_selector = False

        return True

    return False


# ============================================================================
# UI COMPONENTS
# ============================================================================


def display_tables_info():
    """Display information about loaded tables"""
    if st.session_state.files_db and st.session_state.files_db.tables_info:
        common_workbook_bytes = getattr(
            st.session_state.files_db,
            "common_workbook_bytes",
            None,
        )
        if common_workbook_bytes:
            st.download_button(
                "Download Common Flat-File Workbook",
                data=common_workbook_bytes,
                file_name="loaded_files_common_flat_file.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_loaded_common_workbook",
            )
            st.caption(
                "Contains one tab per loaded table and a generated Schema tab "
                "used by the SQL agent."
            )

        # Display overview metrics
        col1, col2, col3 = st.columns(3)
       
        total_tables = len(st.session_state.files_db.tables_info)
        total_rows = sum(info['row_count'] for info in st.session_state.files_db.tables_info.values())
        total_columns = sum(len(info['columns']) for info in st.session_state.files_db.tables_info.values())
       
        with col1:
            st.metric(
                label="📊 Tables Loaded",
                value=total_tables,
                delta=None
            )
       
        with col2:
            st.metric(
                label="📝 Total Rows",
                value=f"{total_rows:,}",
                delta=None
            )
       
        with col3:
            st.metric(
                label="🔢 Total Columns",
                value=total_columns,
                delta=None
            )
       
        st.markdown("---")
        st.subheader("📋 Table Summary")
       
        summary_df = st.session_state.files_db.get_tables_summary()
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Table": st.column_config.TextColumn("Table Name", width="medium"),
                "Source": st.column_config.TextColumn("Source File", width="large"),
                "Rows": st.column_config.NumberColumn("Rows", format="%d"),
                "Columns": st.column_config.NumberColumn("Columns", format="%d"),
            }
        )
       
        st.markdown("---")
       
        # Show expandable details for each table
        for table_name, info in st.session_state.files_db.tables_info.items():
            with st.expander(f"📄 **{table_name}** - {info['row_count']:,} rows"):
                col_a, col_b = st.columns([1, 2])
               
                with col_a:
                    st.markdown("**📂 Source Information**")
                    st.write(f"**File:** {info['source_file']}")
                    if 'source_sheet' in info:
                        st.write(f"**Sheet:** {info['source_sheet']}")
                    st.write(f"**Type:** {info.get('source_type', 'Excel')}")
               
                with col_b:
                    st.markdown("**📊 Column Information**")
                    st.write(f"**Total Columns:** {len(info['columns'])}")
                    st.write(f"**Column Names:** {', '.join(info['columns'][:10])}")
                    if len(info['columns']) > 10:
                        st.write(f"*... and {len(info['columns']) - 10} more columns*")
               
                # Show data preview
                try:
                    preview_query = f"SELECT * FROM {_quote_ident(table_name)} LIMIT 5"
                    preview_df = st.session_state.files_db.execute_query(preview_query)
                    st.markdown("**🔍 Data Preview (first 5 rows):**")
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not load preview: {e}")

                st.markdown("---")
                st.caption("Use advanced cleanup to rename columns, split hierarchy values, and re-apply cleaned data to this in-memory table.")
                show_cleanup = st.checkbox(
                    "Open advanced pandas cleanup",
                    key=f"adv_cleanup_toggle_{table_name}",
                    value=False,
                )
                if show_cleanup:
                    try:
                        full_df = st.session_state.files_db.execute_query(
                            f"SELECT * FROM {_quote_ident(table_name)}"
                        )
                        final_df = render_advanced_table_preview(
                            full_df,
                            file_stem=table_name,
                            key_prefix=f"adv_cleanup_{table_name}",
                        )
                        if final_df is not None and not final_df.empty:
                            apply_col, info_col = st.columns([1, 2])
                            with apply_col:
                                if st.button(
                                    "Apply cleaned table",
                                    key=f"apply_cleaned_{table_name}",
                                    type="primary",
                                    use_container_width=True,
                                ):
                                    orchestrator = st.session_state.agent_orchestrator
                                    if orchestrator is None:
                                        orchestrator = SQLAgentOrchestrator(
                                            st.session_state.azure_client,
                                            st.session_state.azure_config.deployment_name,
                                        )
                                        orchestrator.set_memory(
                                            st.session_state.graph_memory
                                        )
                                        st.session_state.agent_orchestrator = orchestrator
                                    replaced_table = orchestrator.replace_table(
                                        st.session_state.files_db,
                                        table_name,
                                        final_df,
                                        str(info.get('source_file', 'unknown')),
                                        str(info.get('source_sheet', table_name)),
                                        sql_agent=st.session_state.sql_agent,
                                    )
                                    if replaced_table:
                                        st.success(f"✅ Updated table: {replaced_table} ({len(final_df)} rows)")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to update table.")
                            with info_col:
                                st.info("This updates the active in-memory table used for querying in this session.")
                    except Exception as e:
                        st.warning(f"Could not open advanced cleanup for '{table_name}': {e}")


def display_query_interface():
    """Display the query interface"""
    # Welcome message
    st.markdown("""
    <div style='background-color: #f8fafc;
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 4px solid #3b82f6;
                margin-bottom: 2rem;'>
        <h3 style='margin: 0; color: #1e3a8a;'>💬 Ask Your Question</h3>
        <p style='margin: 0.5rem 0 0 0; color: #64748b;'>
            Type your question in natural language and let AI generate the SQL query for you
        </p>
    </div>
    """, unsafe_allow_html=True)
   
    # Example questions in a nicer format
    with st.expander("💡 Need inspiration? Click here for example questions"):
        col1, col2 = st.columns(2)
       
        with col1:
            st.markdown("**📊 Data Exploration:**")
            st.markdown("""
            - How many rows are in each table?
            - Show me all unique values in [column_name]
            - Display the first 20 records
            """)
       
        with col2:
            st.markdown("**📈 Analysis & Aggregation:**")
            st.markdown("""
            - What is the total sum of [column_name]?
            - Calculate the average [column_name] by [group_column]
            - Show me the top 10 records by [column_name]
            """)
   
    files_db = st.session_state.files_db
    pending_clarified = str(st.session_state.get("pending_clarified_question", "") or "").strip()
    if pending_clarified:
        st.session_state.pending_clarified_question = ""
        st.session_state.user_question = pending_clarified
        execute_query(pending_clarified)
        return

    # Graph-level reasoning (planning, sanity checks, cross-question memory)
    # is embedded by default; the opt-out lives in the sidebar settings.
    langgraph_available, _ = SQLAgentOrchestrator.langgraph_status()
    graph_engine = "LangGraph" if langgraph_available else "Local graph"
    memory_count = len(st.session_state.graph_memory)
    if st.session_state.get("use_graph_orchestration", True):
        st.caption(f"🧠 Graph reasoning on · {graph_engine} · {memory_count} memory record(s)")
    else:
        st.caption("🧠 Graph reasoning off — re-enable it in the sidebar to reuse verified results across questions.")

    submit_button = False
    user_question = ""

    # Free-text natural-language question (the only query mode).
    user_question = st.text_area(
        "Your Question:",
        placeholder="e.g., What is the total cost broken down by department?",
        height=100,
        key="user_question",
        help="Type your question in natural language. Be as specific as possible for better results."
    )
    if files_db is not None:
        _display_schema_route_preview(files_db, user_question)

    # Action buttons with better layout
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        submit_button = st.button("🔍 Generate", type="primary", use_container_width=True,)
    with col2:
        clear_button = st.button("🗑️ Clear History", use_container_width=True,)
   
    if clear_button:
        st.session_state.query_history = []
        st.session_state.graph_memory = []
        st.session_state.sql_agent.conversation_history = []
        if st.session_state.agent_orchestrator is not None:
            st.session_state.agent_orchestrator.clear_memory()
        st.success("✨ History cleared successfully!")
        st.rerun()
   
    if submit_button and user_question:
        execute_query(user_question)


def execute_query(user_question: str):
    """Execute a query and display results"""
    with st.spinner("Generating SQL and executing query..."):
        orchestrator = st.session_state.agent_orchestrator
        if orchestrator is None:
            orchestrator = SQLAgentOrchestrator(
                st.session_state.azure_client,
                st.session_state.azure_config.deployment_name,
            )
            orchestrator.set_memory(st.session_state.graph_memory)
            st.session_state.agent_orchestrator = orchestrator
        result = orchestrator.run_free_text_query(
            st.session_state.sql_agent,
            st.session_state.files_db,
            user_question,
            graph_memory=st.session_state.graph_memory,
            use_langgraph=True,
            enable_memory=bool(st.session_state.use_graph_orchestration),
            thread_id=st.session_state.graph_thread_id,
        )
        if st.session_state.use_graph_orchestration:
            st.session_state.graph_memory = list(result.get("graph_memory") or [])
       
        # Add to history
        st.session_state.query_history.insert(0, result)
       
        # Display result
        display_query_result(result)


def display_graph_diagnostics(result: dict):
    """Show compact graph/memory context metadata without exposing hidden reasoning."""
    graph_mode = result.get("graph_mode")
    prompt_tokens = int(result.get("prompt_context_est_tokens") or 0)
    schema_tokens = int(result.get("schema_context_est_tokens") or 0)
    memory_tokens = int(result.get("memory_context_est_tokens") or 0)
    examples_tokens = int(result.get("verified_examples_context_est_tokens") or 0)
    memory_used = result.get("memory_used") or []
    examples_used = result.get("verified_examples_used") or []
    selected_tables = result.get("selected_schema_tables") or []
    schema_confidence = result.get("schema_confidence")
    schema_expanded = bool(result.get("schema_expanded"))
    result_sanity = result.get("result_sanity") or {}

    if graph_mode or prompt_tokens or schema_tokens or memory_tokens or examples_tokens:
        parts = []
        if graph_mode:
            parts.append(f"Graph: {graph_mode}")
        if schema_confidence:
            parts.append(f"schema confidence: {schema_confidence}")
        if schema_expanded:
            parts.append("expanded route")
        if prompt_tokens:
            parts.append(f"prompt context ~{prompt_tokens} tokens")
        if schema_tokens:
            parts.append(f"schema ~{schema_tokens}")
        if memory_tokens:
            parts.append(f"memory ~{memory_tokens}")
        if examples_tokens:
            parts.append(f"examples ~{examples_tokens}")
        if memory_used:
            parts.append(f"{len(memory_used)} memory record(s) used")
        if examples_used:
            parts.append(f"{len(examples_used)} verified example(s)")
        if parts:
            st.caption(" · ".join(parts))

    graph_trace = result.get("graph_trace") or []
    if graph_trace or selected_tables or result_sanity:
        with st.expander("🧭 Graph trace", expanded=False):
            if selected_tables:
                st.caption("Selected schema tables: " + ", ".join(selected_tables))
            if result_sanity:
                flags = result_sanity.get("flags") or []
                st.caption(
                    f"Result sanity: {result_sanity.get('status', 'unknown')}"
                    + (f" ({', '.join(flags)})" if flags else "")
                )
            for line in graph_trace:
                st.text(line)
            if memory_used:
                st.markdown("**Memory used:**")
                for item in memory_used:
                    st.caption(str(item.get("question", "")))
            if examples_used:
                st.markdown("**Verified examples used:**")
                for item in examples_used:
                    st.caption(str(item.get("question", "")))


def display_query_result(result: dict):
    """Display the result of a query (v3: clarification | success | failure)."""
    result_container = st.container()

    with result_container:
        st.markdown("---")

        # ── Clarification needed ───────────────────────────────────────────
        if result.get('needs_clarification'):
            st.markdown("**Your Question:**")
            st.info(result['question'])
            display_graph_diagnostics(result)

            st.markdown(
                "<div style='background:#fffbeb; border-left:4px solid #f59e0b; "
                "padding:1.25rem 1.5rem; border-radius:6px; margin:1rem 0;'>"
                "<div style='font-size:1.05rem; font-weight:700; color:#92400e; "
                "margin-bottom:0.6rem;'>🤔 Clarification needed</div>"
                "<div style='color:#78350f; font-size:1rem;'>"
                + html.escape(str(result.get('clarification_question', 'Could you clarify your question?')))
                + "</div></div>",
                unsafe_allow_html=True,
            )

            options = result.get('clarification_options', [])
            labels  = result.get('option_labels', [])
            if options:
                st.markdown("**Suggested interpretations:**")
                for i, opt in enumerate(options, 1):
                    label = labels[i - 1] if i - 1 < len(labels) and labels[i - 1] else None
                    if label:
                        st.markdown(f"&nbsp;&nbsp;**{i}. {label}:** {opt}")
                    else:
                        st.markdown(f"&nbsp;&nbsp;**{i}.** {opt}")
                    button_label = f"Use option {i}" + (f": {label}" if label else "")
                    button_key = "clarify_" + hashlib.sha256(
                        f"{result.get('question', '')}|{i}|{opt}".encode()
                    ).hexdigest()[:12]
                    if st.button(button_label, key=button_key, use_container_width=True):
                        st.session_state.pending_clarified_question = _refined_question_from_clarification(
                            result,
                            str(opt),
                            str(label or ""),
                        )
                        st.rerun()

            secondary_note = result.get('secondary_note', '')
            if secondary_note:
                st.info(f"💡 {secondary_note}")

            reason = result.get('clarification_reason', '')
            if reason:
                st.caption(f"ℹ️ {reason}")

            st.caption("Or edit the question manually above and run it again.")
            return

        # ── Success ───────────────────────────────────────────────────────
        if result['success']:
            warnings_list = result.get('warnings', [])
            has_financial_guardrail = any(
                _warning_is_financial_guardrail(w) for w in warnings_list
            )
            if has_financial_guardrail:
                st.warning("⚠️ Query ran with financial guardrail warnings.")
            else:
                st.success("✅ Query executed successfully!")


            # Question
            st.markdown("**Your Question:**")
            st.info(result['question'])
            display_graph_diagnostics(result)

            if warnings_list:
                if has_financial_guardrail:
                    st.error("Review financial guardrail warnings before trusting this result.")
                with st.expander(
                    f"⚠️ {len(warnings_list)} warning(s)",
                    expanded=has_financial_guardrail,
                ):
                    for w in warnings_list:
                        st.warning(w)

            # ── Answer summary (new — most prominent) ─────────────────────
            answer_summary = result.get('answer_summary', '')
            if answer_summary:
                safe_answer_summary = html.escape(str(answer_summary)).replace("\n", "<br>")
                st.markdown("**💡 Answer:**")
                st.markdown(
                    f"<div style='background:#f0fdf4; border-left:4px solid #22c55e; "
                    f"padding:1rem; border-radius:6px; margin-bottom:1rem;'>"
                    f"{safe_answer_summary}</div>",
                    unsafe_allow_html=True,
                )

            # ── Repair notice ──────────────────────────────────────────────
            repair_attempts = result.get('repair_attempts', 0)
            if repair_attempts:
                st.info(f"🔧 SQL was repaired automatically ({repair_attempts} attempt(s)).")


            # ── Generated SQL ──────────────────────────────────────────────
            with st.expander("🔍 View Generated SQL", expanded=False):
                st.code(result.get('sql_query', ''), language='sql')


            # ── Debug trace ───────────────────────────────────────────────
            trace = result.get('trace', [])
            if trace:
                with st.expander("🔎 Pipeline trace (debug)", expanded=False):
                    for line in trace:
                        st.text(line)


            # ── Results table ─────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Results")
            results_df = result['results']


            if results_df is not None and len(results_df) > 0:
                # Single-value result → show as metric
                if len(results_df) == 1 and len(results_df.columns) == 1:
                    value = results_df.iloc[0, 0]
                    col_name = results_df.columns[0]
                    try:
                        if isinstance(value, float):
                            formatted = f"{value:,.2f}"
                        elif isinstance(value, int):
                            formatted = f"{value:,}"
                        else:
                            formatted = str(value)
                    except Exception:
                        formatted = str(value)
                    st.markdown(f"**{col_name}:** {formatted}")
                elif len(results_df) <= 20:
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400,
                    )


                # Download
                st.markdown("---")
                csv = results_df.to_csv(index=False)
                col_dl, _ = st.columns([1, 3])
                with col_dl:
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"query_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            else:
                st.warning("⚠️ No results found for this query.")


        else:
            # ── Failure path ───────────────────────────────────────────────
            st.error("❌ Query execution failed")


            st.markdown("**Your Question:**")
            st.info(result['question'])
            display_graph_diagnostics(result)


            # Error message
            with st.expander("🔍 Error Details", expanded=True):
                st.error(result.get('error', 'Unknown error'))


            # SQL that was attempted (if any)
            if result.get('sql_query'):
                with st.expander("🔍 Last SQL Attempted", expanded=False):
                    st.code(result['sql_query'], language='sql')


            # Warnings
            warnings_list = result.get('warnings', [])
            if warnings_list:
                has_financial_guardrail = any(
                    _warning_is_financial_guardrail(w) for w in warnings_list
                )
                with st.expander("⚠️ Warnings", expanded=has_financial_guardrail):
                    for w in warnings_list:
                        st.warning(w)


            # Trace
            trace = result.get('trace', [])
            if trace:
                with st.expander("🔎 Pipeline trace (debug)", expanded=False):
                    for line in trace:
                        st.text(line)


def display_query_history():
    """Display query history (v3: handles clarification, success, and failure entries)."""
    if st.session_state.query_history:
        st.markdown(f"### 📜 Query History ({len(st.session_state.query_history)} queries)")
        st.markdown("---")

        for idx, result in enumerate(st.session_state.query_history):
            # Choose icon based on result type
            if result.get('needs_clarification'):
                status_icon = "🤔"
            elif result['success']:
                status_icon = "✅"
            else:
                status_icon = "❌"

            question_preview = result['question'][:60]
            suffix           = "..." if len(result['question']) > 60 else ""

            with st.expander(
                f"{status_icon} Query {idx + 1}: {question_preview}{suffix}",
                expanded=False,
            ):
                st.markdown(f"**Full Question:** {result['question']}")

                # ── Clarification entry ─────────────────────────────────
                if result.get('needs_clarification'):
                    st.markdown(
                        "<div style='background:#fffbeb; border-left:4px solid #f59e0b; "
                        "padding:0.75rem 1rem; border-radius:6px; margin:0.5rem 0;'>"
                        "<b>🤔 Clarification was requested:</b><br>"
                        + result.get('clarification_question', '') + "</div>",
                        unsafe_allow_html=True,
                    )
                    options = result.get('clarification_options', [])
                    labels  = result.get('option_labels', [])
                    if options:
                        parts = []
                        for i, opt in enumerate(options, 1):
                            label = labels[i - 1] if i - 1 < len(labels) and labels[i - 1] else None
                            parts.append(f"**{i}. {label}:** {opt}" if label else f"**{i}.** {opt}")
                        st.markdown("**Options offered:** " + " · ".join(parts))
                    secondary_note = result.get('secondary_note', '')
                    if secondary_note:
                        st.caption(f"💡 {secondary_note}")

                # ── Success entry ───────────────────────────────────────
                elif result['success']:
                    answer_summary = result.get('answer_summary', '')
                    if answer_summary:
                        st.markdown("**💡 Answer:**")
                        st.markdown(answer_summary)

                    st.markdown("**Generated SQL:**")
                    st.code(result.get('sql_query', ''), language='sql')

                    if result.get('repair_attempts', 0):
                        st.info(f"🔧 SQL was repaired ({result['repair_attempts']} attempt(s)).")

                    results_df = result.get('results')
                    if results_df is not None and len(results_df) > 0:
                        st.markdown("**Results:**")
                        if len(results_df) <= 5:
                            st.dataframe(
                                results_df,
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.dataframe(
                                results_df,
                                use_container_width=True,
                                hide_index=True,
                                height=300,
                            )

                        st.caption(f"📊 {len(results_df)} rows returned")

                    warnings_list = result.get('warnings', [])
                    if warnings_list:
                        with st.expander("⚠️ Warnings", expanded=False):
                            for w in warnings_list:
                                st.warning(w)

                # ── Failure entry ───────────────────────────────────────
                else:
                    st.error(f"**Error:** {result.get('error', 'Unknown error')}")


    else:
        st.markdown(
            """
            <div style='text-align:center; padding:3rem; background-color:#f9fafb;
                        border-radius:15px; border:2px dashed #d1d5db;'>
                <h3 style='color:#6b7280;'>📜 No queries yet</h3>
                <p style='color:#9ca3af;'>Your query history will appear here
                after you run your first query</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# ADVANCED TABLE RENDERING
# ============================================================================

def render_advanced_table_preview(df: pd.DataFrame, file_stem: str, key_prefix: str):
    """Render advanced table preview with full pandas cleanup and data operations.

    This function provides all the functionality from flat_file_builder.py for:
    - Column deletion
    - Column renaming
    - Column splitting
    - Type conversions
    - Hierarchy splitting on | and - delimiters
    - Whitespace trimming
    - Column name cleaning
    - Duplicate column removal
    - Note row filtering
    - Excel download export
    """
    # Use a key that includes the file_stem so all widget state
    # automatically resets when the user switches to a different file/sheet.
    scope = hashlib.sha256(f"{key_prefix}_{file_stem}".encode()).hexdigest()[:8]
    kp = f"{key_prefix}_{scope}"

    st.markdown("### Advanced Table Operations")
    with st.expander("Edit the table with pandas-style operations", expanded=False):
        cleanup_cols = list(df.columns)
        drop_columns = st.multiselect(
            "Delete columns",
            cleanup_cols,
            help="Remove columns from the final output.",
            key=f"{kp}_drop_columns",
        )

        remaining_for_rename = [c for c in cleanup_cols if c not in drop_columns]
        rename_df = pd.DataFrame({
            "Column": remaining_for_rename,
            "New Name": remaining_for_rename,
        })
        edited_rename_df = st.data_editor(
            rename_df,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key=f"{kp}_rename_columns_editor",
        )
        rename_map = {
            row["Column"]: row["New Name"]
            for _, row in edited_rename_df.iterrows()
            if row["Column"] != row["New Name"]
        }

        st.markdown("**Split a column**")
        split_enabled = st.checkbox(
            "Split one column into multiple columns",
            key=f"{kp}_split_enabled",
        )
        split_config = {}
        if split_enabled and remaining_for_rename:
            col_split_a, col_split_b, col_split_c = st.columns(3)
            with col_split_a:
                split_column = st.selectbox(
                    "Column to split",
                    remaining_for_rename,
                    key=f"{kp}_split_column",
                )
            with col_split_b:
                delimiter = st.text_input(
                    "Delimiter",
                    value=" | ",
                    key=f"{kp}_split_delimiter",
                )
            with col_split_c:
                max_parts = st.number_input(
                    "Number of output columns",
                    min_value=2,
                    max_value=12,
                    value=2,
                    key=f"{kp}_split_max_parts",
                )
            col_prefix, col_keep = st.columns([2, 1])
            with col_prefix:
                split_prefix = st.text_input(
                    "Output column prefix",
                    value=clean_column_name(split_column),
                    key=f"{kp}_split_prefix",
                )
            with col_keep:
                keep_original = st.checkbox(
                    "Keep original",
                    value=True,
                    key=f"{kp}_split_keep_original",
                )
            split_config = {
                "column": split_column,
                "delimiter": delimiter,
                "max_parts": max_parts,
                "prefix": split_prefix,
                "keep_original": keep_original,
            }

        drop_blank_columns = []
        type_conversions = {}

        col_strip, col_clean, col_dash = st.columns(3)
        with col_strip:
            strip_text = st.checkbox(
                "Trim whitespace in text columns",
                value=True,
                key=f"{kp}_strip_text",
            )
        with col_clean:
            clean_names = st.checkbox(
                "Clean final column names",
                value=False,
                key=f"{kp}_clean_names",
            )
        with col_dash:
            dash_split_mode = st.selectbox(
                "Auto-split '-' mode",
                options=["Off", "Spaced only ( - )", "Any dash (-)"],
                index=1,
                key=f"{kp}_dash_split_mode",
                help="Controls automatic dash splitting in hierarchy columns.",
            )
        dash_split_mode_map = {
            "Off": "off",
            "Spaced only ( - )": "spaced",
            "Any dash (-)": "any",
        }
        dash_split_mode_value = dash_split_mode_map[dash_split_mode]

    # Cache cleanup result in session_state keyed on inputs so it only
    # recomputes when the user actually changes a cleanup setting.
    cleanup_cache_version = "v2_hierarchy_split_regex"
    cleanup_sig = hashlib.sha256(
        str((
            cleanup_cache_version,
            list(df.columns), len(df),
            sorted(drop_columns),
            sorted(rename_map.items()),
            str(split_config),
            strip_text,
            clean_names,
            dash_split_mode_value,
        )).encode()
    ).hexdigest()
    cleanup_cache_key = f"{kp}_final_df_{cleanup_sig}"

    if cleanup_cache_key not in st.session_state:
        st.session_state[cleanup_cache_key] = apply_pandas_cleanup(
            df,
            drop_columns=drop_columns,
            rename_map=rename_map,
            split_config=split_config,
            drop_blank_columns=drop_blank_columns,
            type_conversions=type_conversions,
            strip_text=strip_text,
            clean_names=clean_names,
            dash_split_mode=dash_split_mode_value,
        )
    final_df = st.session_state[cleanup_cache_key]

    st.markdown("### Final Table Preview")
    col_rows, col_cols = st.columns(2)
    col_rows.metric("Rows", f"{len(final_df):,}")
    col_cols.metric("Columns", f"{len(final_df.columns):,}")
    st.dataframe(final_df.head(200), use_container_width=True)

    # Build Excel bytes only when the user explicitly requests the download.
    # Using a callback keeps the heavy serialization out of every re-render.
    dl_key = f"{kp}_download"
    dl_state_key = f"{kp}_excel_bytes_{cleanup_sig}"  # keyed on cleanup sig so stale bytes are never served

    def _prepare_download():
        st.session_state[dl_state_key] = to_excel_bytes(final_df)

    if dl_state_key not in st.session_state:
        st.button(
            "Prepare Download",
            on_click=_prepare_download,
            use_container_width=True,
            key=f"{kp}_prepare_btn",
        )
    else:
        st.download_button(
            "⬇️ Download Excel",
            data=st.session_state[dl_state_key],
            file_name=f"{file_stem}_processed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=dl_key,
        )

    return final_df


# ============================================================================
# MAIN APP
# ============================================================================


def main():
    """Main application"""
    st.set_page_config(
        page_title="SQL Query Agent - Natural Language to SQL",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
   
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
   
    # Header block
    st.markdown("""
    <div style='background: linear-gradient(to right, #3b82f6, #2563eb);
                padding: 2rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🔍 SQL Query Agent</h1>
        <p style='color: white; opacity: 0.95; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
            Transform your questions into insights with AI-powered SQL generation
        </p>
    </div>
    """, unsafe_allow_html=True)
   
    # Initialize session state
    initialize_session_state()
   
    # Initialize Azure OpenAI
    if not initialize_azure():
        st.stop()
   
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; margin-bottom: 1rem;'>
            <h2 style='color: #1e3a8a; margin: 0;'>📤 Upload Files</h2>
        </div>
        """, unsafe_allow_html=True)
       
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Excel or CSV files",
            type=['xlsx', 'xls', 'csv'],
            accept_multiple_files=True,
            help="Upload one or more Excel (.xlsx, .xls) or CSV (.csv) files to query",
            key="file_uploader"
        )
        uploaded_file_list = uploaded_files if isinstance(uploaded_files, list) else ([uploaded_files] if uploaded_files else [])
       
        # Analyze files and show sheet selection
        if uploaded_file_list:
            # Analyze files to get sheets
            if st.button("📋 Analyze Files", type="secondary", use_container_width=True):
                with st.spinner("Analyzing files..."):
                    file_sheets = analyze_uploaded_files(uploaded_file_list)
                    st.session_state.file_sheets = file_sheets
                    # Initialize selected sheets with all sheets (preserve existing selections when possible)
                    existing_selected = dict(st.session_state.selected_sheets)
                    existing_overrides = dict(st.session_state.flat_file_overrides)
                    st.session_state.selected_sheets = {}
                    st.session_state.flat_file_overrides = {}
                    for file_name, info in file_sheets.items():
                        if info['type'] == 'excel' and info['sheets']:
                            st.session_state.selected_sheets[file_name] = existing_selected.get(file_name, info['sheets'].copy())
                            st.session_state.flat_file_overrides[file_name] = bool(existing_overrides.get(file_name, False))
                    st.session_state.show_sheet_selector = True
                    st.success("✅ Files analyzed! Select sheets below.")
                    st.rerun()
           
            # Show uploaded file names
            st.markdown("**📁 Selected Files:**")
            for uploaded in uploaded_file_list:
                uploaded_name = _uploaded_file_name(uploaded)
                uploaded_bytes = _uploaded_file_bytes(uploaded)
                file_size = len(uploaded_bytes) / (1024 * 1024)  # Convert to MB
                file_extension = os.path.splitext(uploaded_name)[1].lower()
                file_type = "Excel" if file_extension in ['.xlsx', '.xls'] else "CSV"
                st.text(f"• {uploaded_name} ({file_size:.2f} MB, {file_type})")

            # Sheet selection interface
            if st.session_state.show_sheet_selector and st.session_state.file_sheets:
                st.markdown("---")
                st.markdown("**📊 Select Sheets to Load:**")
               
                # Show sheet selection for each Excel file
                has_excel = False
                for file_name, info in st.session_state.file_sheets.items():
                    if info['type'] == 'excel' and info['sheets']:
                        has_excel = True
                        with st.expander(f"📁 {file_name}", expanded=True):
                            st.markdown(f"*Found {len(info['sheets'])} sheet(s)*")
                           
                            # Select all/none buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"✅ Select All", key=f"select_all_{file_name}", use_container_width=True,):
                                    st.session_state.selected_sheets[file_name] = info['sheets'].copy()
                                    # Force widget state update by clearing checkbox keys
                                    for sheet in info['sheets']:
                                        widget_key = f"sheet_{file_name}_{sheet}"
                                        if widget_key in st.session_state:
                                            st.session_state[widget_key] = True
                                    st.rerun()
                            with col2:
                                if st.button(f"❌ Clear All", key=f"clear_all_{file_name}", use_container_width=True,):
                                    st.session_state.selected_sheets[file_name] = []
                                    # Force widget state update by clearing checkbox keys
                                    for sheet in info['sheets']:
                                        widget_key = f"sheet_{file_name}_{sheet}"
                                        if widget_key in st.session_state:
                                            st.session_state[widget_key] = False
                                    st.rerun()
                           
                            # Checkboxes for each sheet with callback
                            def update_sheet_selection(file_name, sheet):
                                """Callback to update sheet selection"""
                                widget_key = f"sheet_{file_name}_{sheet}"
                                is_checked = st.session_state.get(widget_key, False)
                               
                                if file_name not in st.session_state.selected_sheets:
                                    st.session_state.selected_sheets[file_name] = []
                               
                                if is_checked and sheet not in st.session_state.selected_sheets[file_name]:
                                    st.session_state.selected_sheets[file_name].append(sheet)
                                elif not is_checked and sheet in st.session_state.selected_sheets[file_name]:
                                    st.session_state.selected_sheets[file_name].remove(sheet)
                           
                            for sheet in info['sheets']:
                                is_selected = sheet in st.session_state.selected_sheets.get(file_name, [])
                                st.checkbox(
                                    sheet,
                                    value=is_selected,
                                    key=f"sheet_{file_name}_{sheet}",
                                    on_change=update_sheet_selection,
                                    args=(file_name, sheet)
                                )

                            st.markdown("---")
                            st.caption("Workbook processing mode")
                            mode_key = f"processing_mode_{file_name}"
                            current_override = bool(st.session_state.flat_file_overrides.get(file_name, False))
                            mode_options = [
                                "Auto extract report/matrix",
                                "Already flat (skip extraction)",
                            ]
                            default_idx = 1 if current_override else 0
                            selected_mode = st.radio(
                                "How should this workbook be processed?",
                                options=mode_options,
                                index=default_idx,
                                key=mode_key,
                                help="This is separate from sheet selection. Choose 'Already flat' only when tabs are clean table layouts.",
                            )
                            st.session_state.flat_file_overrides[file_name] = (selected_mode == mode_options[1])

                if not has_excel:
                    st.info("ℹ️ No Excel files to configure. CSV files will be loaded automatically.")
               
                # Flat file conversion is always enabled
                st.markdown("---")
                st.info("ℹ️ Excel sheets are auto-flattened by default. You can mark specific files as already flat to skip extraction.")

                # Load button
                st.markdown("---")
                total_selected = sum(len(sheets) for sheets in st.session_state.selected_sheets.values())
                action_label = "Add / Load Files" if st.session_state.files_loaded else "Load Files"
                if st.button(f"🔄 {action_label} ({total_selected} sheet(s) selected)", type="primary", use_container_width=True):
                    with st.spinner("Loading files..."):
                        if load_uploaded_files(
                            uploaded_file_list,
                            st.session_state.selected_sheets,
                            st.session_state.flat_file_overrides,
                        ):
                            st.success("✅ Files loaded successfully!")
                            st.rerun()

        # Clear files button (only show if files are loaded)
        if st.session_state.files_loaded:
            st.markdown("---")
            if st.button("🗑️ Clear All Files", use_container_width=True):
                st.session_state.files_db = None
                st.session_state.sql_agent = None
                st.session_state.query_history = []
                st.session_state.graph_memory = []
                if st.session_state.agent_orchestrator is not None:
                    st.session_state.agent_orchestrator.clear_memory()
                st.session_state.uploaded_files = []
                st.session_state.files_loaded = False
                st.session_state.file_sheets = {}
                st.session_state.selected_sheets = {}
                st.session_state.flat_file_overrides = {}
                st.session_state.show_sheet_selector = False
                st.success("✨ Files cleared! Upload new files to continue.")
                st.rerun()
       
        st.markdown("---")
       
        st.markdown("""
        <div style='text-align: center; padding: 1rem; margin-bottom: 1rem;'>
            <h2 style='color: #1e3a8a; margin: 0;'>ℹ️ About</h2>
        </div>
        """, unsafe_allow_html=True)
       
        st.markdown("""
        <div style='background-color: white;
                    padding: 1.5rem;
                    border-radius: 10px;
                    border: 1px solid #e5e7eb;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);'>
            <p style='margin: 0;'>
                Query Excel and CSV files using natural language - no SQL knowledge required!
            </p>
        </div>
        """, unsafe_allow_html=True)
       
        st.markdown("<br>", unsafe_allow_html=True)
       
        st.markdown("**✨ Key Features:**")
        st.markdown("""
        - 📊 Upload your files & select specific sheets
        - 📋 Support for Excel & CSV
        - 💬 Natural language queries
        - 🔍 AI-powered SQL generation
        - 📥 Export results to CSV
        - 📜 Query history tracking
        """)
       
        st.markdown("---")
       
        # Status section with metrics
        st.markdown("**📊 System Status:**")
        if st.session_state.initialized:
            st.success("✅ Azure OpenAI Ready")
        else:
            st.warning("⚠️ Initializing...")
        langgraph_available, _ = SQLAgentOrchestrator.langgraph_status()
        if langgraph_available:
            st.success("✅ LangGraph Ready")
        else:
            st.info("ℹ️ Local graph fallback active")
        st.toggle(
            "Graph reasoning & memory",
            key="use_graph_orchestration",
            help=(
                "Embedded by default: plans queries as a graph, sanity-checks "
                "results, and reuses verified answers across questions. "
                "Disable only to run each question in isolation."
            ),
        )
       
        if st.session_state.files_loaded and st.session_state.files_db and st.session_state.files_db.tables_info:
            st.success("✅ Files Loaded")
           
            # Display quick stats
            total_tables = len(st.session_state.files_db.tables_info)
            total_queries = len(st.session_state.query_history)
            files_loaded = len(st.session_state.files_db.loaded_files)
           
            st.info(f"📁 Active: {files_loaded} file(s)")
           
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tables", total_tables)
            with col2:
                st.metric("Queries", total_queries)
        else:
            st.info("ℹ️ No files loaded yet")
       
        st.markdown("---")
        st.caption("💡 Tip: Be specific in your questions for better results!")
   
    # Main content with tabs
    if not st.session_state.files_loaded:
        # Show welcome message when no files are loaded
        st.markdown("""
        <div style='text-align: center;
                    padding: 4rem 2rem;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 20px;
                    margin: 2rem 0;
                    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);'>
            <h2 style='color: white; font-size: 2.5rem; margin: 0 0 1rem 0;'>👋 Welcome to SQL Query Agent</h2>
            <p style='color: white; font-size: 1.3rem; opacity: 0.95; margin: 0;'>
                Get started by uploading your Excel or CSV files using the sidebar
            </p>
        </div>
        """, unsafe_allow_html=True)
       
        # Instructions
        col1, col2, col3 = st.columns(3)
       
        with col1:
            st.markdown("""
            <div style='text-align: center; padding: 2rem; background-color: #f8fafc; border-radius: 15px; height: 100%;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>📤</div>
                <h3 style='color: #1e3a8a;'>1. Upload Files</h3>
                <p style='color: #64748b;'>Select Excel (.xlsx, .xls) or CSV files from your computer</p>
            </div>
            """, unsafe_allow_html=True)
       
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem; background-color: #f8fafc; border-radius: 15px; height: 100%;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>📊</div>
                <h3 style='color: #1e3a8a;'>2. Select Sheets</h3>
                <p style='color: #64748b;'>Choose which Excel sheets to process</p>
            </div>
            """, unsafe_allow_html=True)
       
        with col3:
            st.markdown("""
            <div style='text-align: center; padding: 2rem; background-color: #f8fafc; border-radius: 15px; height: 100%;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>💬</div>
                <h3 style='color: #1e3a8a;'>3. Ask & Analyze</h3>
                <p style='color: #64748b;'>Get instant insights with natural language</p>
            </div>
            """, unsafe_allow_html=True)
       
        st.markdown("<br>", unsafe_allow_html=True)
       
        # Additional info
        with st.expander("📖 How to use this application", expanded=False):
            st.markdown("""
            ### Getting Started
           
            1. **Upload Your Data Files**
               - Click the file uploader in the sidebar
               - Select one or more Excel or CSV files
               - Click "Analyze Files" to scan them
           
            2. **Select Sheets (for Excel files)**
               - View all available sheets from your Excel files
               - Select or deselect specific sheets to load
               - Use "Select All" / "Clear All" for quick selection
               - CSV files are automatically included
           
            3. **Choose Processing Options**
               - Selected sheets are automatically checked for flat, report-block, or matrix structure
               - Already-flat sheets bypass expensive extraction
               - Complex sheets are flattened while remaining separate logical tables

            4. **Load Your Data**
               - Click "Load Files" with your selected sheets
               - Generated schema metadata is supplied directly to the AI
               - A common Excel workbook containing all data tables and the Schema tab becomes available
               - Wait for the files to be processed
           
            5. **Ask Questions in Natural Language**
               - Go to the "Ask Questions" tab
               - Type your question like: "What is the total sales by region?"
               - Click "Generate" to execute
           
            6. **View and Export Results**
               - Results appear as tables
               - Download results as CSV
               - View query history anytime
           
            ### Flat File Conversion
            The integrated flat-file builder automatically handles:
            - Complex layouts or merged cells
            - Matrix formats (data in both rows and columns)
            - Unstructured data that's hard to query
            - Existing flat tables, which are loaded without report extraction
           
            ### Supported File Types
            - Excel files (.xlsx, .xls) - choose specific sheets
            - CSV files (.csv) - loaded automatically
           
            ### Example Questions
            - "Show me the first 10 rows"
            - "What is the total revenue by product?"
            - "Calculate the average price by category"
            - "List all unique customer names"
            """)
    else:
        # Show main tabs when files are loaded
        tab1, tab2, tab3 = st.tabs([
            "🔍 Ask Questions",
            "📊 View Tables",
            "📜 Query History"
        ])
       
        with tab1:
            display_query_interface()
       
        with tab2:
            display_tables_info()
       
        with tab3:
            display_query_history()


if __name__ == "__main__":
    main()
