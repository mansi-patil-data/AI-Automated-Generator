"""Tests for business data loading and normalization."""

from io import BytesIO

import pandas as pd
import pytest

from src.dataloader import DataValidationError, load_business_data


def test_load_csv_normalizes_columns_and_drops_duplicate_rows() -> None:
    csv_data = (
        b"Order Date,Sales Region,Product Name,Net Sales,Total Expenses,Units Sold\n"
        b"2025-01-01,East,Widget,100,60,2\n"
        b"2025-01-01,East,Widget,100,60,2\n"
    )
    result = load_business_data(BytesIO(csv_data), "sales.csv")

    assert list(result.columns) == ["date", "region", "product", "revenue", "expenses", "sales_quantity"]
    assert len(result) == 1
    assert result.attrs["source_rows"] == 2
    assert result.attrs["duplicates_removed"] == 1
    assert result.loc[0, "revenue"] == 100
    assert result.loc[0, "date"] == pd.Timestamp("2025-01-01")


def test_invalid_numeric_cells_become_missing_values() -> None:
    result = load_business_data(
        BytesIO(b"Revenue\nnot-a-number\n25\n"),
        "business.csv",
    )
    assert result["revenue"].isna().sum() == 1
    assert result["revenue"].iloc[1] == 25


def test_rejects_unsupported_file_extension() -> None:
    with pytest.raises(DataValidationError, match="Unsupported file type"):
        load_business_data(b"hello", "business.json")


def test_rejects_file_without_business_metrics() -> None:
    with pytest.raises(DataValidationError, match="No recognized business metric"):
        load_business_data(BytesIO(b"Customer,City\nA,London\n"), "business.csv")


def test_excel_upload_is_supported() -> None:
    output = BytesIO()
    pd.DataFrame({"Sales": [12.5], "Territory": ["West"]}).to_excel(output, index=False)
    result = load_business_data(BytesIO(output.getvalue()), "business.xlsx")
    assert result.loc[0, "revenue"] == 12.5
    assert result.loc[0, "region"] == "West"


def test_category_columns_are_normalized() -> None:
    result = load_business_data(
        BytesIO(b"Revenue,Product Category\n12.5,Software\n"),
        "business.csv",
    )
    assert result.loc[0, "category"] == "Software"
