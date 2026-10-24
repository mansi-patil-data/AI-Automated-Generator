"""Tests for KPI and trend calculations."""

import pandas as pd
import pytest

from src.dataanalyzer import analyze_data


@pytest.fixture
def sales_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-18", "2025-02-03"]),
            "region": ["East", "West", "East"],
            "product": ["A", "B", "A"],
            "revenue": [100.0, 50.0, 200.0],
            "expenses": [60.0, 30.0, 120.0],
            "profit": [40.0, 20.0, 80.0],
        }
    )


def test_calculates_totals_margin_and_average_order_value(sales_data: pd.DataFrame) -> None:
    result = analyze_data(sales_data)
    assert result["totals"]["revenue"] == 350
    assert result["totals"]["expenses"] == 210
    assert result["totals"]["profit"] == 140
    assert result["totals"]["profit_margin"] == pytest.approx(40)
    assert result["totals"]["average_order_value"] == pytest.approx(350 / 3)


def test_ranks_regions_and_products(sales_data: pd.DataFrame) -> None:
    result = analyze_data(sales_data)
    assert result["best_region"] == {"name": "East", "revenue": 300}
    assert result["worst_region"] == {"name": "West", "revenue": 50}
    assert result["best_product"] == {"name": "A", "revenue": 300}


def test_calculates_month_over_month_growth(sales_data: pd.DataFrame) -> None:
    result = analyze_data(sales_data)
    assert result["totals"]["mom_growth"] == pytest.approx(100 / 3)
    assert result["monthly_trends"] == [
        {"month": "2025-01", "revenue": 150, "profit": 60},
        {"month": "2025-02", "revenue": 200, "profit": 80},
    ]


def test_zero_previous_month_revenue_returns_no_growth() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "revenue": [0.0, 10.0],
            "expenses": [0.0, 4.0],
        }
    )
    result = analyze_data(data)
    assert result["totals"]["mom_growth"] is None


def test_missing_revenue_and_expenses_are_handled_without_division_by_zero() -> None:
    result = analyze_data(pd.DataFrame({"profit": [4.0, 6.0]}))
    assert result["totals"]["revenue"] == 0
    assert result["totals"]["profit"] == 10
    assert result["totals"]["profit_margin"] == 0
    assert result["totals"]["average_order_value"] == 0


def test_profit_is_not_inferred_when_expenses_are_missing() -> None:
    result = analyze_data(pd.DataFrame({"revenue": [100.0, 50.0]}))
    assert result["totals"]["profit"] == 0
    assert result["availability"]["profit"] is False
    assert result["availability"]["expenses"] is False


def test_profit_is_derived_when_revenue_and_expenses_are_available() -> None:
    result = analyze_data(pd.DataFrame({"revenue": [100.0], "expenses": [65.0]}))
    assert result["totals"]["profit"] == 35
    assert result["availability"]["profit"] is True
    assert result["availability"]["profit_margin"] is True


def test_profit_uses_only_rows_with_both_financial_values() -> None:
    result = analyze_data(
        pd.DataFrame(
            {
                "revenue": [100.0, 900.0],
                "expenses": [60.0, None],
            }
        )
    )
    assert result["totals"]["profit"] == 40
    assert result["availability"]["profit"] is True


def test_profit_is_unavailable_when_financial_rows_do_not_overlap() -> None:
    result = analyze_data(
        pd.DataFrame(
            {
                "revenue": [100.0, None],
                "expenses": [None, 60.0],
            }
        )
    )
    assert result["availability"]["profit"] is False


def test_category_performance_uses_filtered_revenue() -> None:
    result = analyze_data(
        pd.DataFrame(
            {
                "category": ["Software", "Services", "Software"],
                "revenue": [100.0, 80.0, 50.0],
            }
        )
    )
    assert result["category_performance"] == [
        {"name": "Software", "revenue": 150},
        {"name": "Services", "revenue": 80},
    ]
