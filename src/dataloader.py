"""Load and validate tabular business exports."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when an uploaded dataset cannot be analyzed safely."""


ALIASES = {
    "date": {"date", "orderdate", "transactiondate", "salesdate", "datetime"},
    "region": {"region", "salesregion", "territory", "market", "location"},
    "product": {"product", "productname", "item", "itemname", "sku"},
    "category": {"category", "productcategory", "itemcategory", "segment"},
    "revenue": {"revenue", "sales", "totalrevenue", "netsales", "amount"},
    "expenses": {"expenses", "expense", "cost", "costs", "totalexpenses"},
    "profit": {"profit", "netprofit", "grossprofit"},
    "sales_quantity": {
        "salesquantity",
        "quantity",
        "units",
        "unitssold",
        "quantitysold",
        "volume",
    },
}
NUMERIC_COLUMNS = ("revenue", "expenses", "profit", "sales_quantity")


def _normalized_token(column: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(column).strip().lower())


def _canonical_name(column: object) -> str:
    token = _normalized_token(column)
    for canonical, aliases in ALIASES.items():
        if token in aliases:
            return canonical
    name = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
    return name or "unnamed_column"


def _read_table(source: str | Path | bytes | bytearray | BinaryIO, filename: str | None) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise DataValidationError("Unsupported file type. Upload a CSV or Excel file.")

    if filename is None:
        raise DataValidationError("A filename is required when loading file content.")
    suffix = Path(filename).suffix.lower()
    stream = BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else source
    if hasattr(stream, "seek"):
        stream.seek(0)
    if suffix == ".csv":
        return pd.read_csv(stream)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(stream)
    raise DataValidationError("Unsupported file type. Upload a CSV or Excel file.")


def load_business_data(
    source: str | Path | bytes | bytearray | BinaryIO,
    filename: str | None = None,
) -> pd.DataFrame:
    """Read a CSV/Excel file and return validated, normalized business data.

    Known business fields are mapped to canonical names. Duplicate rows are
    removed, invalid dates become ``NaT``, and unparseable numeric cells become
    ``NaN`` so downstream aggregations can skip them safely.
    """
    try:
        data = _read_table(source, filename)
    except DataValidationError:
        raise
    except (OSError, ValueError, ImportError) as exc:
        raise DataValidationError(f"Unable to read the uploaded file: {exc}") from exc

    if data.empty:
        raise DataValidationError("The uploaded file contains no data rows.")
    if len(data.columns) == 0:
        raise DataValidationError("The uploaded file contains no columns.")

    normalized_names = [_canonical_name(column) for column in data.columns]
    normalized = pd.DataFrame(index=data.index)
    for source_position, target_name in enumerate(normalized_names):
        column = data.iloc[:, source_position]
        if target_name in normalized.columns:
            normalized[target_name] = normalized[target_name].combine_first(column)
        else:
            normalized[target_name] = column

    available_metrics = [column for column in NUMERIC_COLUMNS if column in normalized.columns]
    if not available_metrics:
        raise DataValidationError(
            "No recognized business metric was found. Include revenue, expenses, "
            "profit, or sales quantity."
        )

    for column in available_metrics:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized[column] = normalized[column].replace([np.inf, -np.inf], np.nan)
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")

    source_rows = len(normalized)
    duplicates_removed = int(normalized.duplicated().sum())
    normalized = normalized.drop_duplicates().reset_index(drop=True)
    if normalized.empty:
        raise DataValidationError("No rows remain after duplicate removal.")
    if not normalized[available_metrics].notna().any(axis=None):
        raise DataValidationError("All values in the recognized business metrics are invalid.")
    normalized.attrs["source_rows"] = source_rows
    normalized.attrs["duplicates_removed"] = duplicates_removed
    return normalized
