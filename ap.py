import streamlit as st
import os
import re
from uuid import uuid4
from pathlib import Path
import io
from datetime import date, datetime
from dotenv import load_dotenv
import hashlib
from sqlalchemy import create_engine
from typing import Any, cast

# Load .env from the same directory as THIS file — must happen before any
# azure.identity import so env vars are in the process environment in time.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH, override=True)

from sql_agent_core import FilesDatabaseManager, FilesSQLAgent, initialize_azure_client
import pandas as pd
from flat_file_builder import (
    extract_and_flatten_sheet,
    finalize_extracted_sheet,
    apply_pandas_cleanup,
    clean_column_name,
    to_excel_bytes,
    cell_text,
    _skip_error_heavy_rows,
    auto_flatten_report_tables,
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

def _extract_sheet_benelux_safe(file_bytes: bytes, sheet_name: str, preferred_profile: str = "auto") -> pd.DataFrame:
    """Extract sheet with explicit benelux tab support.
    
    This function ensures benelux-style sheets with #REF! errors are properly handled
    by applying error-aware extraction logic from flat_file_builder.
    """
    try:
        # Primary path: use the fixed extract_and_flatten_sheet which handles benelux tabs
        df = extract_and_flatten_sheet(file_bytes, sheet_name, preferred_profile=preferred_profile)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"[benelux_safe] Primary extraction failed: {e}")
    
    # Fallback: try the general auto_flatten approach directly
    try:
        quick_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=None)
        # Skip error-heavy rows for benelux support
        quick_df = _skip_error_heavy_rows(quick_df, max_check=50)
        if quick_df is not None and not quick_df.empty:
            df = auto_flatten_report_tables(quick_df, extraction_profile=preferred_profile)
            if df is not None and not df.empty:
                return finalize_extracted_sheet(df, strip_text=True, split_hierarchy=True)
    except Exception as e:
        print(f"[benelux_safe] Fallback extraction failed: {e}")
    
    # Final fallback: simple flat read with error token awareness
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
        if df is not None and not df.empty:
            # Apply finalize which uses cell_text() to clean error tokens
            return finalize_extracted_sheet(df, strip_text=True, split_hierarchy=True)
    except Exception as e:
        print(f"[benelux_safe] Final fallback failed: {e}")
    
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
        st.session_state.files_db = None
        st.session_state.sql_agent = None
        st.session_state.query_history = []
        st.session_state.uploaded_files = []
        st.session_state.files_loaded = False
        st.session_state.file_sheets = {}  # Maps file names to their sheets
        st.session_state.selected_sheets = {}  # Maps file names to selected sheets
        st.session_state.flat_file_overrides = {}  # Maps Excel file names to 'already flat' boolean
        st.session_state.show_sheet_selector = False
        st.session_state.convert_to_flat = True  # Always convert to flat format

    # Backfill keys for existing sessions after app updates.
    if 'flat_file_overrides' not in st.session_state:
        st.session_state.flat_file_overrides = {}


# ============================================================================
# INITIALIZATION
# ============================================================================


def get_excel_sheets(file_path):
    """Extract sheet names from an Excel file"""
    try:
        excel_file = pd.ExcelFile(file_path)
        sheets = excel_file.sheet_names
        excel_file.close()
        return sheets
    except Exception as e:
        st.error(f"Error reading sheets from {os.path.basename(file_path)}: {e}")
        return []


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


def _safe_file_stem(file_name: str) -> str:
    stem = Path(file_name).stem or "upload"
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)


def _save_uploaded_file_to_temp(uploaded_file, temp_dir: str) -> str:
    """Persist an uploaded file to a unique temp path to avoid locked-file collisions."""
    os.makedirs(temp_dir, exist_ok=True)
    file_name = _uploaded_file_name(uploaded_file)
    ext = Path(file_name).suffix.lower() or ".bin"
    stem = _safe_file_stem(file_name)
    unique_name = f"{stem}_{uuid4().hex[:10]}{ext}"
    temp_file_path = os.path.join(temp_dir, unique_name)
    with open(temp_file_path, 'wb') as f:
        f.write(_uploaded_file_bytes(uploaded_file))
    return temp_file_path


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

    Keeps processing minimal to avoid expensive hierarchy/matrix transforms.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    cleaned = pd.DataFrame(df).copy()
    text_cols = [
        col for col in cleaned.columns
        if pd.api.types.is_object_dtype(cleaned[col].dtype) or pd.api.types.is_string_dtype(cleaned[col].dtype)
    ]
    if text_cols:
        cleaned[text_cols] = cleaned[text_cols].apply(lambda s: s.astype(str).str.strip())

    # Drop columns that are completely blank/null after trim.
    cols_to_keep = []
    for col in cleaned.columns:
        series = cleaned[col]
        if series.isna().all():
            continue
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            lowered = series.astype(str).str.strip().str.lower()
            if lowered.isin(["", "nan", "none"]).all():
                continue
        cols_to_keep.append(col)

    return pd.DataFrame(cleaned.loc[:, cols_to_keep])




def convert_excel_to_flat_format_deterministic(file_path, sheet_names):
    """Auto-detect + extract selected sheets into separate flat tabs."""
    try:
        structured_sheets = {}
        file_bytes = Path(file_path).read_bytes()

        def _safe_excel_sheet_name(name: str, used_names: set[str]) -> str:
            # Excel sheet names cannot contain these characters and are max 31 chars.
            cleaned = str(name)
            for bad in [":", "\\", "/", "?", "*", "[", "]"]:
                cleaned = cleaned.replace(bad, "_")
            cleaned = cleaned.strip() or "sheet"
            cleaned = cleaned[:31]

            if cleaned not in used_names:
                used_names.add(cleaned)
                return cleaned

            base = cleaned[:28]
            suffix = 2
            while True:
                candidate = f"{base}_{suffix}"[:31]
                if candidate not in used_names:
                    used_names.add(candidate)
                    return candidate
                suffix += 1

        for sheet_name in sheet_names:
            st.write(f"🔄 Flattening sheet (deterministic): {sheet_name}")

            try:
                structured_df = _extract_sheet_benelux_safe(file_bytes, sheet_name, preferred_profile="auto")
            except Exception as e:
                st.warning(f"⚠️ Deterministic extraction failed for '{sheet_name}', using raw read. ({e})")
                structured_df = pd.DataFrame()

            if structured_df is None or structured_df.empty:
                fallback_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
                if fallback_df.empty:
                    st.warning(f"⚠️ Sheet '{sheet_name}' is empty, skipping...")
                    continue
                structured_df = pd.DataFrame(fallback_df)
                structured_df = finalize_extracted_sheet(structured_df, strip_text=True, split_hierarchy=True)

            structured_df = _apply_default_pandas_cleanup(pd.DataFrame(structured_df))
            structured_sheets[sheet_name] = structured_df
            st.success(f"✅ Processed sheet: {sheet_name} ({len(structured_df)} rows)")

        if structured_sheets:
            flattened_sheets = {}
            used_sheet_names = set()

            # Keep this path lightweight: extraction already normalizes structure.
            for source_sheet, raw_df in structured_sheets.items():
                cleaned_df = pd.DataFrame(raw_df)
                if cleaned_df is None or cleaned_df.empty:
                    st.warning(f"⚠️ Flattened output for '{source_sheet}' has no rows, skipping...")
                    continue

                cleaned_df = _apply_default_pandas_cleanup(cleaned_df)

                output_sheet_name = _safe_excel_sheet_name(source_sheet, used_sheet_names)
                flattened_sheets[output_sheet_name] = cleaned_df

            if not flattened_sheets:
                st.warning("⚠️ Deterministic conversion produced no flat rows. Using original workbook.")
                return file_path, sheet_names

            temp_dir = os.path.join(os.getcwd(), 'temp_uploads')
            os.makedirs(temp_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            safe_base = re.sub(r"[^A-Za-z0-9._-]", "_", base_name)
            output_path = os.path.join(temp_dir, f"{safe_base}_flat_{uuid4().hex[:8]}.xlsx")

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for out_sheet_name, out_df in flattened_sheets.items():
                    out_df.to_excel(writer, sheet_name=out_sheet_name, index=False)

            st.success(
                f"✅ Converted file saved: {os.path.basename(output_path)} "
                f"({len(flattened_sheets)} flattened tab(s), kept separate)"
            )
            return output_path, list(flattened_sheets.keys())

        return file_path, sheet_names

    except Exception as e:
        st.error(f"❌ Error in deterministic flat file conversion: {e}")
        return file_path, sheet_names


def _load_dataframe_into_files_db(
    files_db: FilesDatabaseManager,
    df: pd.DataFrame,
    source_file_name: str,
    source_sheet_name: str,
) -> str:
    """Load an in-memory dataframe as a table inside FilesDatabaseManager."""

    def _coerce_sqlite_value(value):
        # SQLite driver does not bind pandas.Timestamp reliably from object columns.
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

    if files_db.engine is None:
        files_db.engine = create_engine(
            'sqlite:///:memory:',
            connect_args={'check_same_thread': False}
        )
        files_db.connection = files_db.engine.connect()

    working_df = cast(pd.DataFrame, pd.DataFrame(df).copy())
    if working_df.empty:
        return ""

    cleaned_columns: list[str] = [str(files_db._clean_column_name(c)) for c in list(working_df.columns)]
    working_df.columns = cleaned_columns
    table_name = str(files_db._clean_table_name(source_sheet_name or "sheet"))
    if not table_name:
        return ""

    # Normalize datetime/object columns so sqlite bindings are always supported.
    for col in working_df.columns:
        series = working_df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            dt = pd.to_datetime(series, errors="coerce")
            working_df[col] = dt.dt.strftime("%Y-%m-%d %H:%M:%S").where(~dt.isna(), None)
            continue

        if series.dtype == object:
            working_df[col] = series.map(_coerce_sqlite_value)

    working_df.to_sql(table_name, files_db.connection, if_exists='replace', index=False)
    column_types = {str(col): str(dtype) for col, dtype in working_df.dtypes.items()}
    columns_list: list[str] = list(cleaned_columns)
    row_count = int(working_df.shape[0])
    files_db.tables_info[table_name] = {
        'source_file': source_file_name,
        'source_sheet': source_sheet_name,
        'columns': columns_list,
        'row_count': row_count,
        'column_types': column_types,
    }
    return table_name


def _replace_table_in_files_db(
    files_db: FilesDatabaseManager,
    table_name: str,
    df: pd.DataFrame,
    source_file_name: str,
    source_sheet_name: str,
) -> str:
    """Replace an existing in-memory table with cleaned dataframe content."""

    def _coerce_sqlite_value(value):
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

    if files_db.engine is None:
        files_db.engine = create_engine(
            'sqlite:///:memory:',
            connect_args={'check_same_thread': False}
        )
        files_db.connection = files_db.engine.connect()

    working_df = cast(pd.DataFrame, pd.DataFrame(df).copy())
    if working_df.empty:
        return ""

    cleaned_columns: list[str] = [str(files_db._clean_column_name(c)) for c in list(working_df.columns)]
    working_df.columns = cleaned_columns

    for col in working_df.columns:
        series = working_df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            dt = pd.to_datetime(series, errors="coerce")
            working_df[col] = dt.dt.strftime("%Y-%m-%d %H:%M:%S").where(~dt.isna(), None)
            continue
        if series.dtype == object:
            working_df[col] = series.map(_coerce_sqlite_value)

    target_table = str(files_db._clean_table_name(table_name))
    if not target_table:
        return ""

    working_df.to_sql(target_table, files_db.connection, if_exists='replace', index=False)
    files_db.tables_info[target_table] = {
        'source_file': source_file_name,
        'source_sheet': source_sheet_name,
        'columns': list(cleaned_columns),
        'row_count': int(working_df.shape[0]),
        'column_types': {str(col): str(dtype) for col, dtype in working_df.dtypes.items()},
    }
    return target_table


def load_data(
    uploaded_files: list[Any] | None,
    selected_sheets=None,
    convert_to_flat=False,
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

            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), 'temp_uploads')
            os.makedirs(temp_dir, exist_ok=True)
           
            for uploaded_file in uploaded_files:
                file_name = _uploaded_file_name(uploaded_file)
                file_bytes = _uploaded_file_bytes(uploaded_file)
                file_extension = os.path.splitext(file_name)[1].lower()
                force_flat = bool(flat_file_overrides and flat_file_overrides.get(file_name, False))

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
                            table_name = _load_dataframe_into_files_db(
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
                if convert_to_flat and file_extension in ['.xlsx', '.xls'] and not force_flat:
                    if selected_sheets and file_name in selected_sheets:
                        sheets_to_process = selected_sheets[file_name]
                    else:
                        sheets_to_process = get_excel_sheets_from_bytes(file_bytes)

                    if not sheets_to_process:
                        st.warning(f"⚠️ No sheets selected/found for: {file_name}")
                        continue

                    preloaded_sheet_fallbacks = _read_excel_sheets_from_bytes(file_bytes, sheets_to_process)

                    prepared_sheets: dict[str, pd.DataFrame] = {}
                    failed_sheet_names: list[str] = []

                    # Phase 1: extract/prepare so users can see what parsed successfully.
                    for sheet_name in sheets_to_process:
                        st.write(f"🔄 {sheet_name}: extracting report/matrix blocks")

                        try:
                            structured_df = _extract_sheet_benelux_safe(
                                file_bytes,
                                sheet_name,
                                preferred_profile="auto",
                            )
                        except Exception as e:
                            st.warning(f"⚠️ Extraction failed for '{sheet_name}', using raw read. ({e})")
                            structured_df = pd.DataFrame()

                        if structured_df is None or structured_df.empty:
                            fallback_df = preloaded_sheet_fallbacks.get(sheet_name, pd.DataFrame())
                            if fallback_df is None or fallback_df.empty:
                                fallback_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
                            if fallback_df is None or fallback_df.empty:
                                failed_sheet_names.append(sheet_name)
                                st.warning(f"⚠️ Sheet '{sheet_name}' is empty, skipping...")
                                continue
                            structured_df = pd.DataFrame(fallback_df)
                            structured_df = finalize_extracted_sheet(structured_df, strip_text=True, split_hierarchy=True)

                        structured_df = _apply_default_pandas_cleanup(pd.DataFrame(structured_df))
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
                            table_name = _load_dataframe_into_files_db(
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

                # Legacy path for non-flatten mode: persist once and load from disk.
                temp_file_path = _save_uploaded_file_to_temp(uploaded_file, temp_dir)

                # Apply flat file conversion if enabled (only for Excel files)
                converted_sheet_names = None
                if convert_to_flat and file_extension in ['.xlsx', '.xls']:
                    if selected_sheets and file_name in selected_sheets:
                        sheets_to_convert = selected_sheets[file_name]
                    else:
                        sheets_to_convert = get_excel_sheets(temp_file_path)

                    if sheets_to_convert:
                        st.info(f"🔄 Processing {len(sheets_to_convert)} sheet(s) with auto flat builder...")
                        # Process ALL selected sheets via the new extractor path.
                        # Flat sheets are fast-pathed inside extract_and_flatten_sheet.
                        temp_file_path, converted_sheet_names = convert_excel_to_flat_format_deterministic(
                            temp_file_path,
                            sheets_to_convert,
                        )

                # Check if we have sheet selections for this file
                sheets_to_load = None
                if converted_sheet_names:
                    sheets_to_load = converted_sheet_names
                elif selected_sheets and file_name in selected_sheets:
                    sheets_to_load = selected_sheets[file_name]

                # Load the file with sheet selection
                if file_extension in ['.xlsx', '.xls']:
                    # noinspection SqlNoDataSourceInspection
                    if files_db.load_excel_file(temp_file_path, sheets_to_load):
                        loaded_count += 1
                        sheet_info = f" ({len(sheets_to_load)} sheet(s))" if sheets_to_load else ""
                        st.success(f"✅ Loaded: {file_name}{sheet_info}")
                    else:
                        st.warning(f"⚠️ Failed to load: {file_name}")
                elif file_extension == '.csv':
                    if files_db.load_csv_file(temp_file_path):
                        loaded_count += 1
                        st.success(f"✅ Loaded: {file_name}")
                    else:
                        st.warning(f"⚠️ Failed to load: {file_name}")

            if loaded_count > 0:
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
                st.session_state.initialized = True
                return True
            except Exception as e:
                st.error(f"Failed to initialize Azure OpenAI client: {e}")
                return False


    return True




def load_uploaded_files(uploaded_files, selected_sheets=None, flat_file_overrides=None):
    """Load uploaded files and initialize SQL agent"""
    if uploaded_files:
        existing_db = st.session_state.files_db if st.session_state.files_loaded else None
        files_db = load_data(
            uploaded_files,
            selected_sheets,
            st.session_state.convert_to_flat,
            existing_files_db=existing_db,
            flat_file_overrides=flat_file_overrides,
        )

        if files_db is None or not files_db.tables_info:
            st.error("No data was loaded. Please check your files.")
            return False

        sql_agent = FilesSQLAgent(
            st.session_state.azure_client,
            files_db,
            st.session_state.azure_config.deployment_name
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
            width="stretch",
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
                    preview_query = f"SELECT * FROM {table_name} LIMIT 5"
                    preview_df = st.session_state.files_db.execute_query(preview_query)
                    st.markdown("**🔍 Data Preview (first 5 rows):**")
                    st.dataframe(preview_df, width="stretch", hide_index=True)
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
                        full_df = st.session_state.files_db.execute_query(f"SELECT * FROM {table_name}")
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
                                    width="stretch",
                                ):
                                    replaced_table = _replace_table_in_files_db(
                                        st.session_state.files_db,
                                        table_name,
                                        final_df,
                                        str(info.get('source_file', 'unknown')),
                                        str(info.get('source_sheet', table_name)),
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
    tables_info = files_db.tables_info if files_db and files_db.tables_info else {}

    query_mode = st.radio(
        "Query mode",
        options=["Guided", "Free text"],
        horizontal=True,
        key="query_mode_selector",
    )

    submit_button = False
    user_question = ""
    guided_payload = None

    if query_mode == "Guided":
        st.caption("Use guided mode for deterministic query templates (faster and safer for common analysis).")
        guided_intent = st.selectbox(
            "What do you want to do?",
            options=["Dataset overview", "Breakdown by tab"],
            key="guided_intent",
        )

        if guided_intent == "Dataset overview":
            user_question = "Show dataset overview across loaded tables"
            guided_payload = {
                "intent": "overview",
                "question": user_question,
            }
        elif guided_intent == "Breakdown by tab":
            sample_values_per_column = int(
                st.number_input(
                    "Sample values per column",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key="guided_tab_breakdown_samples",
                )
            )
            user_question = "Show column/type/value breakdown for each loaded table"
            guided_payload = {
                "intent": "tab_breakdown",
                "question": user_question,
                "sample_values_per_column": sample_values_per_column,
            }

    else:
        # Question input with better styling
        user_question = st.text_area(
            "Your Question:",
            placeholder="e.g., What is the total cost broken down by department?",
            height=100,
            key="user_question",
            help="Type your question in natural language. Be as specific as possible for better results."
        )

    # Action buttons with better layout
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        submit_button = st.button("🔍 Generate", type="primary", width="stretch",)
    with col2:
        clear_button = st.button("🗑️ Clear History", width="stretch",)
   
    if clear_button:
        st.session_state.query_history = []
        st.session_state.sql_agent.conversation_history = []
        st.success("✨ History cleared successfully!")
        st.rerun()
   
    if submit_button:
        if query_mode == "Guided":
            if guided_payload is None:
                st.warning("Please complete the guided selections before running.")
            else:
                execute_guided_query(guided_payload)
        elif user_question:
            execute_query(user_question)




def execute_query(user_question: str):
    """Execute a query and display results"""
    with st.spinner("Generating SQL and executing query..."):
        result = st.session_state.sql_agent.execute_query_with_explanation(user_question)
       
        # Add to history
        st.session_state.query_history.insert(0, result)
       
        # Display result
        display_query_result(result)


def execute_guided_query(guided_payload: dict):
    """Execute a deterministic guided query and display results."""
    with st.spinner("Running guided query..."):
        result = st.session_state.sql_agent.execute_guided_query(guided_payload)
        st.session_state.query_history.insert(0, result)
        display_query_result(result)

def display_query_result(result: dict):
    """Display the result of a query (v3: clarification | success | failure)."""
    result_container = st.container()

    with result_container:
        st.markdown("---")

        # ── Clarification needed ───────────────────────────────────────────
        if result.get('needs_clarification'):
            st.markdown("**Your Question:**")
            st.info(result['question'])

            st.markdown(
                "<div style='background:#fffbeb; border-left:4px solid #f59e0b; "
                "padding:1.25rem 1.5rem; border-radius:6px; margin:1rem 0;'>"
                "<div style='font-size:1.05rem; font-weight:700; color:#92400e; "
                "margin-bottom:0.6rem;'>🤔 Clarification needed</div>"
                "<div style='color:#78350f; font-size:1rem;'>"
                + result.get('clarification_question', 'Could you clarify your question?')
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

            secondary_note = result.get('secondary_note', '')
            if secondary_note:
                st.info(f"💡 {secondary_note}")

            reason = result.get('clarification_reason', '')
            if reason:
                st.caption(f"ℹ️ {reason}")

            st.markdown(
                "💬 *Please refine your question using one of the options above and try again.*"
            )
            return

        # ── Success ───────────────────────────────────────────────────────
        if result['success']:
            st.success("✅ Query executed successfully!")


            # Question
            st.markdown("**Your Question:**")
            st.info(result['question'])


            # ── Answer summary (new — most prominent) ─────────────────────
            answer_summary = result.get('answer_summary', '')
            if answer_summary:
                st.markdown("**💡 Answer:**")
                st.markdown(
                    f"<div style='background:#f0fdf4; border-left:4px solid #22c55e; "
                    f"padding:1rem; border-radius:6px; margin-bottom:1rem;'>"
                    f"{answer_summary}</div>",
                    unsafe_allow_html=True,
                )


            # ── Warnings ──────────────────────────────────────────────────
            warnings_list = result.get('warnings', [])
            if warnings_list:
                with st.expander(f"⚠️ {len(warnings_list)} warning(s)", expanded=False):
                    for w in warnings_list:
                        st.warning(w)


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
                    st.dataframe(results_df, width="stretch", hide_index=True)
                else:
                    st.dataframe(
                        results_df,
                        width="stretch",
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
                        width="stretch",
                    )
            else:
                st.warning("⚠️ No results found for this query.")


        else:
            # ── Failure path ───────────────────────────────────────────────
            st.error("❌ Query execution failed")


            st.markdown("**Your Question:**")
            st.info(result['question'])


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
                with st.expander("⚠️ Warnings", expanded=False):
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
                                width="stretch",
                                hide_index=True,
                            )
                        else:
                            st.dataframe(
                                results_df,
                                width="stretch",
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
    scope = hashlib.md5(f"{key_prefix}_{file_stem}".encode()).hexdigest()[:8]
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
    cleanup_sig = hashlib.md5(
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
            if st.button("📋 Analyze Files", type="secondary", width="stretch"):
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
                                if st.button(f"✅ Select All", key=f"select_all_{file_name}", width="stretch",):
                                    st.session_state.selected_sheets[file_name] = info['sheets'].copy()
                                    # Force widget state update by clearing checkbox keys
                                    for sheet in info['sheets']:
                                        widget_key = f"sheet_{file_name}_{sheet}"
                                        if widget_key in st.session_state:
                                            st.session_state[widget_key] = True
                                    st.rerun()
                            with col2:
                                if st.button(f"❌ Clear All", key=f"clear_all_{file_name}", width="stretch",):
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
                if st.button(f"🔄 {action_label} ({total_selected} sheet(s) selected)", type="primary", width="stretch"):
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
            if st.button("🗑️ Clear All Files", width="stretch"):
                st.session_state.files_db = None
                st.session_state.sql_agent = None
                st.session_state.query_history = []
                st.session_state.uploaded_files = []
                st.session_state.files_loaded = False
                st.session_state.file_sheets = {}
                st.session_state.selected_sheets = {}
                st.session_state.flat_file_overrides = {}
                st.session_state.show_sheet_selector = False
                st.session_state.convert_to_flat = True
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
               - **Optional**: Enable "Convert selected sheets to flat format"
               - This is useful for Excel files with complex layouts, merged cells, or matrix formats
               - Deterministic flattening will auto-detect each selected tab and merge them into one flat table
           
            4. **Load Your Data**
               - Click "Load Files" with your selected sheets
               - If conversion is enabled, selected sheets are auto-detected, flattened, and merged
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
            Enable this option AFTER selecting sheets if your Excel files have:
            - Complex layouts or merged cells
            - Matrix formats (data in both rows and columns)
            - Unstructured data that's hard to query
           
            The deterministic flattener will auto-detect and reshape selected sheets, then merge them into one flat output table.
           
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

