"""Flat File Builder

Debugging app for turning messy Excel sheets into queryable flat tables.
Run with:
    streamlit run flat_file_builder.py
"""

import io
import re
from datetime import date, datetime
from pathlib import Path

import openpyxl
import pandas as pd
import streamlit as st


def is_blank(value) -> bool:
    return pd.isna(value) or str(value).strip().lower() in {"", "nan", "none"}


def cell_text(value) -> str:
    if is_blank(value):
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def clean_column_name(value) -> str:
    text = str(value).strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    text = text.strip("_") or "unnamed_column"
    if text[0].isdigit():
        text = f"col_{text}"
    return text


def dedupe_columns(columns):
    counts = {}
    result = []
    for col in columns:
        base = clean_column_name(col)
        counts[base] = counts.get(base, 0) + 1
        result.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return result


def primary_excel_format_section(number_format: str) -> str:
    section = str(number_format or "General").split(";")[0]
    section = re.sub(r'"[^"]*"', "", section)
    section = re.sub(r"\[[^\]]+\]", "", section)
    return section


def decimal_places_from_excel_format(number_format: str) -> int | None:
    section = primary_excel_format_section(number_format)
    if "." not in section:
        return 0 if any(token in section for token in ("0", "#", "?")) else None
    decimals = section.split(".", 1)[1]
    decimals = decimals.split("%", 1)[0]
    placeholders = [char for char in decimals if char in {"0", "#", "?"}]
    return len(placeholders) if placeholders else 0


def formatted_excel_value(cell):
    value = cell.value
    if value is None:
        return ""

    if isinstance(value, (datetime, date)):
        return value.strftime("%d.%m.%Y")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number_format = str(cell.number_format or "General")
        decimals = decimal_places_from_excel_format(number_format)
        if decimals is None:
            return value

        is_percent = "%" in primary_excel_format_section(number_format)
        display_value = value * 100 if is_percent else value
        use_grouping = "," in number_format
        show_plus = "+" in number_format and display_value > 0
        sign = "+" if show_plus else ""
        grouped = "," if use_grouping else ""
        rendered = f"{sign}{display_value:{grouped}.{decimals}f}"
        if is_percent:
            rendered = f"{rendered}%"
        return rendered

    return value


def row_has_values(row, start_col=1) -> bool:
    return any(not is_blank(v) for v in row.iloc[start_col:].tolist())


def first_nonblank_after(row, start_col=1) -> str:
    for value in row.iloc[start_col:].tolist():
        if not is_blank(value):
            return cell_text(value)
    return ""


def is_unit_label(value) -> bool:
    text = cell_text(value).lower()
    return text == "%" or text.startswith(("eur ", "usd ", "gbp ", "in %"))


def nonblank_col_indexes(row, start_col=1):
    return [
        idx for idx in range(start_col, len(row))
        if not is_blank(row.iloc[idx])
    ]


def drop_all_blank_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    keep_columns = [
        col for col in df.columns
        if not df[col].apply(is_blank).all()
    ]
    return df.loc[:, keep_columns].copy()


def looks_like_data_value(value) -> bool:
    text = cell_text(value)
    if not text or is_unit_label(text):
        return False
    if metric_context(text)["metric_type"]:
        return False
    cleaned = text.replace(",", "").replace("%", "").replace("+", "").replace("−", "-").strip()
    if cleaned.lower() in {"n.m.", "n.m", "nm"}:
        return True
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def extract_bottom_notes(raw_df: pd.DataFrame) -> list[str]:
    if raw_df is None or raw_df.empty:
        return []

    last_data_row = -1
    for row_idx in range(len(raw_df)):
        values = raw_df.iloc[row_idx].tolist()
        if sum(1 for value in values if looks_like_data_value(value)) >= 2:
            last_data_row = row_idx

    if last_data_row < 0:
        return []

    notes = []
    for row_idx in range(last_data_row + 1, len(raw_df)):
        parts = [cell_text(value) for value in raw_df.iloc[row_idx].tolist() if cell_text(value)]
        if not parts:
            continue
        note = " ".join(parts)
        if is_unit_label(note):
            continue
        notes.append(note)

    return notes


def find_unit_header_cells(raw_df: pd.DataFrame):
    cells = []
    if raw_df is None or raw_df.empty:
        return cells
    unit_prefixes = ("eur ", "usd ", "gbp ", "in %")
    for row_idx in range(len(raw_df)):
        row = raw_df.iloc[row_idx]
        for col_idx, value in enumerate(row.tolist()):
            text = cell_text(value).lower()
            if (is_unit_label(text) or text.startswith(unit_prefixes)) and first_nonblank_after(row, col_idx + 1):
                cells.append((row_idx, col_idx))
    return cells


def section_name(value) -> str:
    text = cell_text(value)
    text = text.replace("¹", "").replace("²", "").replace("³", "")
    return text.strip()


def metric_parse_text(value) -> str:
    text = cell_text(value)
    text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+$", "", text).strip()
    text = re.sub(r"\s+\d+\)$", "", text).strip()
    return text


def carry_forward_header_values(values):
    carried = []
    current = ""
    for value in values:
        if not is_blank(value):
            current = cell_text(value)
        carried.append(current)
    return carried


def auto_flatten_sectioned_financial_sheet(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten stacked financial statement sections into a long analytical table.

    Handles patterns like:
    - Exchange rates / Spot or Average with periods across columns
    - Valuation rates with valuation dates spanning tenor columns
    """
    records = []
    i = 0
    while i < len(raw_df):
        row = raw_df.iloc[i]
        first = cell_text(row.iloc[0])
        first_lower = first.lower()
        rate_label = first_nonblank_after(row, start_col=1)

        if first_lower.startswith("exchange rates") and rate_label:
            rate_type = rate_label
            unit = cell_text(raw_df.iloc[i + 1, 0]) if i + 1 < len(raw_df) else ""
            headers = raw_df.iloc[i + 1].tolist() if i + 1 < len(raw_df) else []
            j = i + 2
            while j < len(raw_df):
                data_row = raw_df.iloc[j]
                label = cell_text(data_row.iloc[0])
                label_lower = label.lower()
                if (
                    not label
                    or label_lower.startswith("exchange rates")
                    or label_lower.startswith("valuation rates")
                    or label_lower.startswith("1)")
                    or label_lower.startswith("2)")
                    or label_lower.startswith("3)")
                ):
                    break
                for col_idx in range(1, len(data_row)):
                    value = data_row.iloc[col_idx]
                    period = cell_text(headers[col_idx]) if col_idx < len(headers) else ""
                    if period and not is_blank(value):
                        records.append({
                            "section": section_name(first),
                            "rate_type": rate_type,
                            "unit": unit,
                            "item": label,
                            "currency": label,
                            "period": period,
                            "valuation_date": "",
                            "tenor": "",
                            "contract_type": "",
                            "value": value,
                        })
                j += 1
            i = j
            continue

        if first_lower.startswith("valuation rates"):
            date_headers = carry_forward_header_values(row.tolist())
            tenor_row = raw_df.iloc[i + 1].tolist() if i + 1 < len(raw_df) else []
            unit = cell_text(tenor_row[0]) if tenor_row else ""
            current_contract = ""
            j = i + 2
            while j < len(raw_df):
                data_row = raw_df.iloc[j]
                label = cell_text(data_row.iloc[0])
                label_lower = label.lower()
                if (
                    not label
                    or label_lower.startswith("valuation rates")
                    or label_lower.startswith("exchange rates")
                    or label_lower.startswith("1)")
                    or label_lower.startswith("2)")
                    or label_lower.startswith("3)")
                ):
                    break

                if not row_has_values(data_row, start_col=1):
                    current_contract = label
                    j += 1
                    continue

                for col_idx in range(1, len(data_row)):
                    value = data_row.iloc[col_idx]
                    valuation_date = date_headers[col_idx] if col_idx < len(date_headers) else ""
                    tenor = cell_text(tenor_row[col_idx]) if col_idx < len(tenor_row) else ""
                    if valuation_date and tenor and not is_blank(value):
                        records.append({
                            "section": section_name(first),
                            "rate_type": "",
                            "unit": unit,
                            "item": label,
                            "currency": label,
                            "period": "",
                            "valuation_date": valuation_date,
                            "tenor": tenor,
                            "contract_type": current_contract,
                            "value": value,
                        })
                j += 1
            i = j
            continue

        i += 1

    return pd.DataFrame(records)


def nearest_title_above(raw_df: pd.DataFrame, row_idx: int) -> str:
    for idx in range(row_idx - 1, -1, -1):
        first = cell_text(raw_df.iloc[idx, 0])
        if not first:
            continue
        if first.lower().startswith(("eur ", "usd ", "gbp ", "in %")):
            continue
        if row_has_values(raw_df.iloc[idx], start_col=1):
            continue
        return first
    return ""


def nearest_group_above(raw_df: pd.DataFrame, row_idx: int) -> str:
    for idx in range(row_idx - 1, -1, -1):
        if cell_text(raw_df.iloc[idx, 0]):
            continue
        group = first_nonblank_after(raw_df.iloc[idx], start_col=1)
        if group:
            return group
    return ""


def nearest_left_label_above(raw_df: pd.DataFrame, row_idx: int, col_idx: int) -> str:
    for idx in range(row_idx - 1, -1, -1):
        label = cell_text(raw_df.iloc[idx, col_idx])
        if label and not label.replace(" ", "").isupper():
            return label
    return ""


def nearest_nonblank_right(raw_df: pd.DataFrame, row_idx: int, col_idx: int, block_end: int) -> str:
    row = raw_df.iloc[row_idx]
    for idx in range(col_idx + 1, min(block_end, len(row))):
        value = cell_text(row.iloc[idx])
        if value:
            return value
    return ""


def cell_indent(raw_df: pd.DataFrame, row_idx: int, col_idx: int) -> float:
    indents = raw_df.attrs.get("excel_indents") or []
    if row_idx >= len(indents) or col_idx >= len(indents[row_idx]):
        return 0
    return indents[row_idx][col_idx] or 0


def indented_row_context(raw_df: pd.DataFrame, row_idx: int, label_col: int) -> dict:
    current_indent = cell_indent(raw_df, row_idx, label_col)
    if current_indent <= 0:
        return {}

    parents = []
    next_indent = current_indent
    for idx in range(row_idx - 1, -1, -1):
        label = cell_text(raw_df.iloc[idx, label_col])
        if not label:
            continue
        indent = cell_indent(raw_df, idx, label_col)
        if indent < next_indent:
            parents.append(label)
            next_indent = indent
            if indent <= 0:
                break

    parents.reverse()
    if not parents:
        return {}

    line_item = cell_text(raw_df.iloc[row_idx, label_col])
    return {
        "parent_line_item": parents[-1],
        "line_item_path": " > ".join([*parents, line_item]),
    }


def metric_context(metric: str) -> dict:
    text = metric_parse_text(metric)
    empty_context = {
        "metric_type": "",
        "metric_date": "",
        "comparison_date": "",
        "metric_year": "",
        "metric_quarter": "",
        "comparison_year": "",
    }

    def normalize_year(year_text: str) -> int:
        year = int(year_text)
        return year + 2000 if len(year_text) == 2 else year

    date_pattern = r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})"
    delta_date_match = re.fullmatch(rf"[∆Δ]\s*{date_pattern}\s*/\s*{date_pattern}", text)
    if delta_date_match:
        start_day, start_month, start_year, end_day, end_month, end_year = delta_date_match.groups()
        return {
            **empty_context,
            "metric_type": "date_delta",
            "metric_date": pd.Timestamp(normalize_year(start_year), int(start_month), int(start_day)),
            "comparison_date": pd.Timestamp(normalize_year(end_year), int(end_month), int(end_day)),
        }

    date_match = re.fullmatch(date_pattern, text)
    if date_match:
        day, month, year = date_match.groups()
        normalized_year = normalize_year(year)
        return {
            **empty_context,
            "metric_type": "date",
            "metric_date": pd.Timestamp(normalized_year, int(month), int(day)),
            "metric_year": normalized_year,
        }

    quarter_match = re.fullmatch(r"([1-4])Q\s*(\d{2,4})", text)
    if quarter_match:
        year_text = quarter_match.group(2)
        year = int(year_text) + 2000 if len(year_text) == 2 else int(year_text)
        return {
            **empty_context,
            "metric_type": "quarter",
            "metric_date": pd.Timestamp(year, int(quarter_match.group(1)) * 3, 1) + pd.offsets.MonthEnd(0),
            "metric_year": year,
            "metric_quarter": int(quarter_match.group(1)),
        }

    year_match = re.fullmatch(r"(20\d{2})", text)
    if year_match:
        year = int(year_match.group(1))
        return {
            **empty_context,
            "metric_type": "year",
            "metric_date": pd.Timestamp(year, 12, 31),
            "metric_year": year,
        }

    delta_match = re.fullmatch(r"[∆Δ]\s*(20\d{2})\s*/\s*(20\d{2})", text)
    if delta_match:
        year = int(delta_match.group(1))
        comparison_year = int(delta_match.group(2))
        return {
            **empty_context,
            "metric_type": "delta",
            "metric_date": pd.Timestamp(year, 12, 31),
            "comparison_date": pd.Timestamp(comparison_year, 12, 31),
            "metric_year": year,
            "comparison_year": comparison_year,
        }

    return empty_context


def header_row_score(row) -> int:
    return sum(
        1
        for value in row.tolist()[1:]
        if cell_text(value) and not metric_context(cell_text(value))["metric_type"] and not is_unit_label(value)
    )


def best_group_header_row(raw_df: pd.DataFrame, unit_row_idx: int) -> list[str]:
    best_idx = None
    best_score = 0
    for idx in range(max(0, unit_row_idx - 4), unit_row_idx):
        score = header_row_score(raw_df.iloc[idx])
        if score > best_score:
            best_idx = idx
            best_score = score
    if best_idx is None:
        return [""] * raw_df.shape[1]
    return carry_forward_header_values(raw_df.iloc[best_idx].tolist())


def best_label_column(raw_df: pd.DataFrame, unit_row_idx: int, unit_cols: list[int]) -> int:
    if not unit_cols:
        return 0

    best_col = 0
    best_score = -1
    for col_idx in range(min(unit_cols)):
        score = 0
        for row_idx in range(unit_row_idx + 1, len(raw_df)):
            row = raw_df.iloc[row_idx]
            label = cell_text(row.iloc[col_idx])
            if not label or is_unit_label(label):
                continue
            if any(not is_blank(row.iloc[value_col]) for value_col in unit_cols):
                score += 1
        if score > best_score:
            best_col = col_idx
            best_score = score

    return best_col


def auto_flatten_grouped_metric_blocks(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten wide report tables with row labels and repeated metric groups."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    records = []
    for unit_row_idx in range(1, len(raw_df)):
        unit_row = raw_df.iloc[unit_row_idx]
        unit_cols = [
            col_idx for col_idx in range(1, raw_df.shape[1])
            if is_unit_label(unit_row.iloc[col_idx])
        ]
        if len(unit_cols) < 4:
            continue

        label_col = best_label_column(raw_df, unit_row_idx, unit_cols)
        period_row = raw_df.iloc[unit_row_idx - 1]
        period_headers = carry_forward_header_values(period_row.tolist())
        detail_headers = raw_df.iloc[unit_row_idx - 2].tolist() if unit_row_idx >= 2 else []
        group_headers = best_group_header_row(raw_df, unit_row_idx)
        table_name = nearest_title_above(raw_df, unit_row_idx)
        if not table_name:
            table_name = first_nonblank_after(raw_df.iloc[max(0, unit_row_idx - 4)], start_col=1)

        row_idx = unit_row_idx + 1
        blank_label_rows = 0
        while row_idx < len(raw_df):
            data_row = raw_df.iloc[row_idx]
            line_item = cell_text(data_row.iloc[label_col])
            if sum(1 for value in data_row.tolist() if is_unit_label(value)) >= 4:
                break
            if not line_item:
                blank_label_rows += 1
                if blank_label_rows >= 5:
                    break
                row_idx += 1
                continue
            blank_label_rows = 0
            if is_unit_label(line_item):
                break
            if all(is_blank(data_row.iloc[col_idx]) for col_idx in unit_cols):
                row_idx += 1
                continue

            row_context = indented_row_context(raw_df, row_idx, label_col)
            for col_idx in unit_cols:
                value = data_row.iloc[col_idx]
                if is_blank(value):
                    continue
                metric = cell_text(period_headers[col_idx])
                if not metric:
                    metric = cell_text(group_headers[col_idx])
                metric_detail = cell_text(detail_headers[col_idx]) if col_idx < len(detail_headers) else ""
                if metric_detail in {metric, cell_text(group_headers[col_idx])}:
                    metric_detail = ""
                record = {
                    "table_name": table_name,
                    "section": table_name,
                    "column_group": cell_text(group_headers[col_idx]),
                    "unit": cell_text(unit_row.iloc[col_idx]),
                    "line_item": line_item,
                    "metric": metric,
                    "metric_detail": metric_detail,
                    "value": value,
                    "block_key": f"{unit_row_idx}:{cell_text(group_headers[col_idx])}",
                    "block_start_column": col_idx,
                }
                record.update(row_context)
                record.update(metric_context(metric))
                records.append(record)
            row_idx += 1

    return pd.DataFrame(records)


def auto_flatten_report_blocks(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Extract repeated side-by-side report blocks into long rows."""
    unit_cells = find_unit_header_cells(raw_df)
    if not unit_cells:
        return pd.DataFrame()

    records = []
    blocks_by_row = {}
    for row_idx, col_idx in unit_cells:
        blocks_by_row.setdefault(row_idx, []).append(col_idx)

    for header_row_idx, starts in blocks_by_row.items():
        starts = sorted(starts)
        for pos, start_col in enumerate(starts):
            end_col = starts[pos + 1] if pos + 1 < len(starts) else raw_df.shape[1]
            header_row = raw_df.iloc[header_row_idx]
            unit = cell_text(header_row.iloc[start_col])
            metric_cols = [
                col for col in range(start_col + 1, end_col)
                if not is_blank(header_row.iloc[col])
            ]
            if not metric_cols:
                continue

            table_name = nearest_left_label_above(raw_df, header_row_idx, start_col)
            section = cell_text(raw_df.iloc[header_row_idx - 1, start_col]) if header_row_idx > 0 else ""
            column_group = nearest_nonblank_right(
                raw_df,
                header_row_idx - 1,
                start_col,
                end_col,
            ) if header_row_idx > 0 else ""

            row_idx = header_row_idx + 1
            blank_label_rows = 0
            while row_idx < len(raw_df):
                data_row = raw_df.iloc[row_idx]
                line_item = cell_text(data_row.iloc[start_col])
                if not line_item:
                    blank_label_rows += 1
                    if blank_label_rows >= 5:
                        break
                    row_idx += 1
                    continue
                blank_label_rows = 0
                if line_item.lower().startswith(("eur ", "usd ", "gbp ", "in %")):
                    break
                if not row_has_values(data_row, start_col=start_col + 1):
                    row_idx += 1
                    continue

                row_context = indented_row_context(raw_df, row_idx, start_col)
                for metric_col in metric_cols:
                    metric = cell_text(header_row.iloc[metric_col])
                    value = data_row.iloc[metric_col]
                    if metric and not is_blank(value):
                        record = {
                            "table_name": table_name,
                            "section": section,
                            "column_group": column_group,
                            "unit": unit,
                            "line_item": line_item,
                            "metric": metric,
                            "value": value,
                            "block_key": f"{header_row_idx}:{start_col}",
                            "block_start_column": start_col,
                        }
                        record.update(row_context)
                        record.update(metric_context(metric))
                        records.append(record)
                row_idx += 1

    return pd.DataFrame(records)


def auto_flatten_report_tables(raw_df: pd.DataFrame) -> pd.DataFrame:
    """General extractor for visually formatted report sheets.

    This is the single user-facing auto mode. Internally it tries a few
    structural strategies and chooses the richest useful output. That keeps the
    UI general without pretending every visual report has one physical shape.
    """
    candidates = [
        ("grouped_metric_blocks", auto_flatten_grouped_metric_blocks(raw_df)),
        ("side_by_side_blocks", auto_flatten_report_blocks(raw_df)),
        ("stacked_tables", auto_flatten_stacked_tables(raw_df)),
        ("sectioned_tables", auto_flatten_sectioned_financial_sheet(raw_df)),
    ]

    best_name = ""
    best_df = pd.DataFrame()
    best_score = 0
    for name, df in candidates:
        if df is None or df.empty:
            continue
        context_cols = [
            col for col in df.columns
            if col not in {"value", "block_start_column"}
        ]
        score = len(df) * max(len(context_cols), 1)
        if score > best_score:
            best_name = name
            best_df = df.copy()
            best_score = score

    if best_df.empty:
        return best_df

    normalized = best_df.copy()

    if "block_id" not in normalized.columns:
        if "block_key" in normalized.columns:
            starts = normalized["block_key"].fillna("-1")
            normalized = normalized.drop(columns=["block_key"])
            normalized.insert(
                1,
                "block_id",
                pd.factorize(starts.astype(str))[0] + 1,
            )
        elif "block_start_column" in normalized.columns:
            starts = normalized["block_start_column"].fillna(-1)
            normalized.insert(
                1,
                "block_id",
                pd.factorize(starts.astype(str))[0] + 1,
            )
        else:
            normalized.insert(1, "block_id", 1)

    if "block_start_column" in normalized.columns:
        normalized = normalized.drop(columns=["block_start_column"])

    if "line_item" not in normalized.columns:
        if "item" in normalized.columns:
            normalized["line_item"] = normalized["item"]
        elif "currency" in normalized.columns:
            normalized["line_item"] = normalized["currency"]
        else:
            normalized["line_item"] = ""

    if "metric" not in normalized.columns:
        metric_parts = []
        for _, row in normalized.iterrows():
            parts = [
                cell_text(row.get("period", "")),
                cell_text(row.get("valuation_date", "")),
                cell_text(row.get("tenor", "")),
                cell_text(row.get("rate_type", "")),
            ]
            metric_parts.append(" | ".join([part for part in parts if part]) or "value")
        normalized["metric"] = metric_parts

    if "table_name" not in normalized.columns:
        normalized["table_name"] = normalized.get("section", "")

    if "unit" not in normalized.columns:
        normalized["unit"] = ""

    for col in normalized.columns:
        if col != "value":
            normalized[col] = normalized[col].fillna("")

    ordered_columns = [
        "table_name",
        "block_id",
        "section",
        "column_group",
        "unit",
        "parent_line_item",
        "line_item_path",
        "line_item",
        "metric",
        "metric_detail",
        "metric_type",
        "metric_date",
        "comparison_date",
        "metric_year",
        "metric_quarter",
        "comparison_year",
        "value",
    ]
    existing_ordered = [col for col in ordered_columns if col in normalized.columns]
    remaining = [col for col in normalized.columns if col not in existing_ordered]
    return drop_all_blank_columns(normalized[existing_ordered + remaining])


def auto_flatten_stacked_tables(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten common stacked mini-table layouts into metric/value rows.

    Looks for header rows where column 0 is a unit label (for example EUR mn)
    and later nonblank cells are metrics/periods. Rows below become line items
    until a run of blank rows or the next header.
    """
    records = []
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    unit_prefixes = ("eur ", "usd ", "gbp ", "in %")
    i = 0
    while i < len(raw_df):
        row = raw_df.iloc[i]
        first = cell_text(row.iloc[0])
        first_lower = first.lower()
        metric_cols = nonblank_col_indexes(row, start_col=1)

        if first_lower.startswith(unit_prefixes) and metric_cols:
            unit = first
            table_name = nearest_title_above(raw_df, i)
            column_group = nearest_group_above(raw_df, i)
            metric_headers = {
                col_idx: cell_text(row.iloc[col_idx])
                for col_idx in metric_cols
            }
            j = i + 1
            blank_label_rows = 0
            while j < len(raw_df):
                data_row = raw_df.iloc[j]
                item = cell_text(data_row.iloc[0])
                item_lower = item.lower()
                if not item:
                    blank_label_rows += 1
                    if blank_label_rows >= 5:
                        break
                    j += 1
                    continue
                blank_label_rows = 0
                if item_lower.startswith(unit_prefixes):
                    break
                if not row_has_values(data_row, start_col=1):
                    j += 1
                    continue

                row_context = indented_row_context(raw_df, j, 0)
                for col_idx, metric in metric_headers.items():
                    value = data_row.iloc[col_idx]
                    if not is_blank(value):
                        record = {
                            "table_name": table_name,
                            "column_group": column_group,
                            "unit": unit,
                            "line_item": item,
                            "metric": metric,
                            "value": value,
                        }
                        record.update(row_context)
                        record.update(metric_context(metric))
                        records.append(record)
                j += 1
            i = j
            continue

        i += 1

    return pd.DataFrame(records)


def read_sheet(uploaded_file, sheet_name, header=None):
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header)


def read_display_sheet(uploaded_file, sheet_name) -> pd.DataFrame:
    """Read an Excel sheet using the values as they are displayed in Excel."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".xlsx", ".xlsm"}:
        return read_sheet(uploaded_file, sheet_name, header=None)

    uploaded_file.seek(0)
    workbook = openpyxl.load_workbook(uploaded_file, data_only=True)
    worksheet = workbook[sheet_name]
    merged_sources = {}
    for merged_range in worksheet.merged_cells.ranges:
        source_cell = worksheet.cell(merged_range.min_row, merged_range.min_col)
        for row_idx in range(merged_range.min_row, merged_range.max_row + 1):
            for col_idx in range(merged_range.min_col, merged_range.max_col + 1):
                merged_sources[(row_idx, col_idx)] = source_cell

    rows = []
    indents = []
    for row in worksheet.iter_rows():
        rows.append([
            formatted_excel_value(merged_sources.get((cell.row, cell.column), cell))
            for cell in row
        ])
        indents.append([
            float(merged_sources.get((cell.row, cell.column), cell).alignment.indent or 0)
            if merged_sources.get((cell.row, cell.column), cell).alignment else 0
            for cell in row
        ])
    df = pd.DataFrame(rows)
    df.attrs["excel_indents"] = indents
    return df


def apply_pandas_cleanup(
    df: pd.DataFrame,
    drop_columns,
    rename_map,
    split_config,
    drop_blank_columns,
    type_conversions,
    strip_text,
    clean_names,
) -> pd.DataFrame:
    cleaned = df.copy()
    if cleaned.empty:
        return cleaned

    if drop_columns:
        cleaned = cleaned.drop(columns=[c for c in drop_columns if c in cleaned.columns])

    if rename_map:
        cleaned = cleaned.rename(columns={
            old: new for old, new in rename_map.items()
            if old in cleaned.columns and new
        })

    if strip_text:
        for col in cleaned.select_dtypes(include="object").columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    if split_config and split_config.get("column") in cleaned.columns:
        source_col = split_config["column"]
        delimiter = split_config.get("delimiter") or " "
        max_parts = int(split_config.get("max_parts") or 2)
        prefix = split_config.get("prefix") or source_col
        keep_original = split_config.get("keep_original", True)
        parts = cleaned[source_col].astype(str).str.split(
            delimiter,
            n=max_parts - 1,
            expand=True,
            regex=False,
        )
        for idx in range(max_parts):
            new_col = clean_column_name(f"{prefix}_{idx + 1}")
            cleaned[new_col] = parts[idx] if idx in parts.columns else ""
        if not keep_original:
            cleaned = cleaned.drop(columns=[source_col])

    if drop_blank_columns:
        present = [c for c in drop_blank_columns if c in cleaned.columns]
        if present:
            mask = cleaned[present].apply(
                lambda row: all(is_blank(value) for value in row),
                axis=1,
            )
            cleaned = cleaned.loc[~mask].copy()

    for col, target_type in (type_conversions or {}).items():
        if col not in cleaned.columns or target_type == "keep":
            continue
        if target_type == "numeric":
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        elif target_type == "datetime":
            cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
        elif target_type == "text":
            cleaned[col] = cleaned[col].astype(str)

    if clean_names:
        cleaned.columns = dedupe_columns(cleaned.columns)

    return drop_all_blank_columns(cleaned).reset_index(drop=True)


def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="flat_table", index=False)
    return output.getvalue()


def excel_mime_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".xls":
        return "application/vnd.ms-excel"
    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def render_raw_excel_download(uploaded_file, key: str):
    st.download_button(
        "Download Raw Excel",
        data=uploaded_file.getvalue(),
        file_name=uploaded_file.name,
        mime=excel_mime_type(uploaded_file.name),
        use_container_width=True,
        key=key,
    )


def render_cleanup_and_preview(df: pd.DataFrame, file_stem: str, key_prefix: str):
    """Apply optional pandas cleanup, then preview and download the result."""
    st.markdown("### Pandas Cleanup")
    with st.expander("Edit the table with pandas-style operations", expanded=False):
        cleanup_cols = list(df.columns)
        drop_columns = st.multiselect(
            "Delete columns",
            cleanup_cols,
            help="Remove columns from the final output.",
            key=f"{key_prefix}_drop_columns",
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
            key=f"{key_prefix}_rename_columns_editor",
        )
        rename_map = {
            row["Column"]: row["New Name"]
            for _, row in edited_rename_df.iterrows()
            if row["Column"] != row["New Name"]
        }

        st.markdown("**Split a column**")
        split_enabled = st.checkbox(
            "Split one column into multiple columns",
            key=f"{key_prefix}_split_enabled",
        )
        split_config = {}
        if split_enabled and remaining_for_rename:
            col_split_a, col_split_b, col_split_c = st.columns(3)
            with col_split_a:
                split_column = st.selectbox(
                    "Column to split",
                    remaining_for_rename,
                    key=f"{key_prefix}_split_column",
                )
            with col_split_b:
                delimiter = st.text_input(
                    "Delimiter",
                    value=" | ",
                    key=f"{key_prefix}_split_delimiter",
                )
            with col_split_c:
                max_parts = st.number_input(
                    "Number of output columns",
                    min_value=2,
                    max_value=12,
                    value=2,
                    key=f"{key_prefix}_split_max_parts",
                )
            col_prefix, col_keep = st.columns([2, 1])
            with col_prefix:
                split_prefix = st.text_input(
                    "Output column prefix",
                    value=clean_column_name(split_column),
                    key=f"{key_prefix}_split_prefix",
                )
            with col_keep:
                keep_original = st.checkbox(
                    "Keep original",
                    value=True,
                    key=f"{key_prefix}_split_keep_original",
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

        col_strip, col_clean = st.columns(2)
        with col_strip:
            strip_text = st.checkbox(
                "Trim whitespace in text columns",
                value=True,
                key=f"{key_prefix}_strip_text",
            )
        with col_clean:
            clean_names = st.checkbox(
                "Clean final column names",
                value=False,
                key=f"{key_prefix}_clean_names",
            )

    final_df = apply_pandas_cleanup(
        df,
        drop_columns=drop_columns,
        rename_map=rename_map,
        split_config=split_config,
        drop_blank_columns=drop_blank_columns,
        type_conversions=type_conversions,
        strip_text=strip_text,
        clean_names=clean_names,
    )

    st.markdown("### Final Table Preview")
    col_rows, col_cols = st.columns(2)
    col_rows.metric("Rows", f"{len(final_df):,}")
    col_cols.metric("Columns", f"{len(final_df.columns):,}")
    st.dataframe(final_df.head(200), use_container_width=True)

    st.download_button(
        "Download Excel",
        data=to_excel_bytes(final_df),
        file_name=f"{file_stem}_flat.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key=f"{key_prefix}_download",
    )


def render_flat_file_tab():
    st.subheader("Already Flat Files")
    st.caption("Use this for clean CSV/Excel tables where each row is already a record and the first row contains headers.")
    uploaded_file = st.file_uploader(
        "Upload a flat CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="flat_file_upload",
    )
    if not uploaded_file:
        st.info("Upload a clean table to preview it.")
        return

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(uploaded_file)
        source_label = Path(uploaded_file.name).stem
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Sheet", xls.sheet_names, key="flat_sheet")
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        source_label = f"{Path(uploaded_file.name).stem}_{sheet_name}"

    st.markdown("### Uploaded Table Preview")
    st.dataframe(df.head(100), use_container_width=True)
    render_cleanup_and_preview(df, source_label, "flat")


def render_report_file_tab():
    st.subheader("General Block Excel Files")
    st.caption("Use this for visually formatted reports with mini-tables, side-by-side blocks, titles, units, and spacer columns.")
    uploaded_file = st.file_uploader(
        "Upload a report-style Excel file",
        type=["xlsx", "xls"],
        key="report_file_upload",
    )
    if not uploaded_file:
        st.info("Upload a formatted report workbook to extract its blocks.")
        return

    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Sheet", xls.sheet_names, key="report_sheet")
    raw_df = read_display_sheet(uploaded_file, sheet_name)

    st.markdown("### Raw Sheet Preview")
    st.dataframe(raw_df.head(40), use_container_width=True)

    extracted_df = auto_flatten_report_tables(raw_df)
    if extracted_df.empty:
        st.warning("No report blocks were detected on this sheet. Try the Already Flat Files tab if this sheet is already a table.")
        render_raw_excel_download(uploaded_file, "report_raw_download_failed")
        return

    required = {"block_id", "line_item", "metric", "value"}
    missing = required - set(extracted_df.columns)
    if missing:
        st.warning(f"Missing expected output columns: {', '.join(sorted(missing))}")

    st.markdown("### Extracted Blocks Preview")
    col_rows, col_cols, col_blocks = st.columns(3)
    col_rows.metric("Rows", f"{len(extracted_df):,}")
    col_cols.metric("Columns", f"{len(extracted_df.columns):,}")
    col_blocks.metric(
        "Blocks",
        f"{extracted_df['block_id'].nunique():,}" if "block_id" in extracted_df.columns else "n/a",
    )
    st.dataframe(extracted_df.head(200), use_container_width=True)

    notes = extract_bottom_notes(raw_df)
    if notes:
        st.markdown("### Notes")
        st.text_area(
            "Extracted sheet notes",
            value="\n".join(notes),
            height=140,
            disabled=True,
            label_visibility="collapsed",
            key="report_extracted_notes",
        )

    source_label = f"{Path(uploaded_file.name).stem}_{sheet_name}"
    render_cleanup_and_preview(extracted_df, source_label, "report")


def main():
    st.set_page_config(
        page_title="Flat File Builder",
        page_icon="🧱",
        layout="wide",
    )
    st.title("Flat File Builder")
    st.caption("Choose the file structure first, then preview and clean the resulting table.")

    flat_tab, report_tab = st.tabs([
        "Already Flat Files",
        "General Block Excel Files",
    ])
    with flat_tab:
        render_flat_file_tab()
    with report_tab:
        render_report_file_tab()


if __name__ == "__main__":
    main()
