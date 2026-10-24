"""Deterministic KPI and performance analysis for normalized business data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _number(value: Any) -> float:
    result = float(value)
    return 0.0 if not np.isfinite(result) else result


def _group_performance(data: pd.DataFrame, dimension: str) -> list[dict[str, Any]]:
    if dimension not in data.columns or "revenue" not in data.columns:
        return []
    grouped = (
        data.groupby(dimension, dropna=True)["revenue"]
        .sum(min_count=1)
        .dropna()
        .sort_values(ascending=False, kind="stable")
    )
    return [{"name": str(name), "revenue": _number(value)} for name, value in grouped.items()]


def _growth(current: float, previous: float) -> float | None:
    if previous == 0 or not np.isfinite(previous):
        return None
    return _number((current - previous) / abs(previous) * 100)


def analyze_data(data: pd.DataFrame) -> dict[str, Any]:
    """Calculate totals, margin, order value, rankings, and monthly trends.

    Rows are treated as orders for average order value because the supported
    schema does not include a distinct order identifier.
    """
    if data.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    revenue = (
        pd.to_numeric(data["revenue"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if "revenue" in data
        else pd.Series(dtype=float)
    )
    expenses = (
        pd.to_numeric(data["expenses"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        if "expenses" in data
        else pd.Series(dtype=float)
    )
    profit_series = (
        pd.to_numeric(data["profit"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        if "profit" in data
        else pd.Series(dtype=float)
    )

    revenue_total = _number(revenue.sum(min_count=1)) if not revenue.empty else 0.0
    expenses_total = _number(expenses.sum(min_count=1)) if not expenses.empty else 0.0
    has_revenue = bool(not revenue.empty and revenue.notna().any())
    has_expenses = bool(not expenses.empty and expenses.notna().any())
    has_explicit_profit = bool(
        not profit_series.empty and profit_series.notna().any()
    )
    has_derived_profit = bool(
        has_revenue
        and has_expenses
        and (revenue.notna() & expenses.notna()).any()
    )
    has_profit = has_explicit_profit or has_derived_profit

    if has_explicit_profit:
        profit_total = _number(profit_series.sum(min_count=1))
    elif has_derived_profit:
        profit_total = _number((revenue - expenses).sum(min_count=1))
    else:
        profit_total = 0.0

    order_count = int(revenue.notna().sum()) if not revenue.empty else 0
    profit_margin = (
        profit_total / revenue_total * 100
        if has_profit and has_revenue and revenue_total
        else 0.0
    )
    average_order_value = revenue_total / order_count if order_count else 0.0

    monthly_trends: list[dict[str, Any]] = []
    mom_growth: float | None = None
    if "date" in data.columns:
        dated = data.loc[pd.to_datetime(data["date"], errors="coerce").notna()].copy()
        if not dated.empty:
            dated["month"] = pd.to_datetime(dated["date"]).dt.to_period("M").dt.to_timestamp()
            aggregates: dict[str, str] = {}
            if has_revenue:
                aggregates["revenue"] = "sum"
            if has_explicit_profit:
                aggregates["profit"] = "sum"
            elif has_derived_profit:
                dated["profit"] = (
                    pd.to_numeric(dated["revenue"], errors="coerce")
                    - pd.to_numeric(dated["expenses"], errors="coerce")
                )
                aggregates["profit"] = "sum"
            if aggregates:
                monthly = dated.groupby("month", as_index=False).agg(aggregates)
                for row in monthly.to_dict(orient="records"):
                    monthly_trends.append(
                        {
                            "month": pd.Timestamp(row["month"]).strftime("%Y-%m"),
                            "revenue": _number(row.get("revenue", 0)),
                            "profit": _number(row.get("profit", 0)),
                        }
                    )
            if len(monthly_trends) >= 2:
                mom_growth = _growth(
                    monthly_trends[-1]["revenue"], monthly_trends[-2]["revenue"]
                )

    region_performance = _group_performance(data, "region")
    product_performance = _group_performance(data, "product")
    category_performance = _group_performance(data, "category")
    return {
        "totals": {
            "revenue": revenue_total,
            "expenses": expenses_total,
            "profit": profit_total,
            "profit_margin": _number(profit_margin),
            "average_order_value": _number(average_order_value),
            "order_count": order_count,
            "mom_growth": mom_growth,
        },
        "best_region": region_performance[0] if region_performance else None,
        "worst_region": region_performance[-1] if region_performance else None,
        "best_product": product_performance[0] if product_performance else None,
        "worst_product": product_performance[-1] if product_performance else None,
        "region_performance": region_performance,
        "product_performance": product_performance,
        "category_performance": category_performance,
        "monthly_trends": monthly_trends,
        "availability": {
            "revenue": has_revenue,
            "expenses": has_expenses,
            "profit": has_profit,
            "profit_margin": has_profit and has_revenue and revenue_total != 0,
            "orders": order_count > 0,
        },
        "data_quality": {
            "rows": int(len(data)),
            "valid_dates": (
                int(pd.to_datetime(data["date"], errors="coerce").notna().sum())
                if "date" in data.columns
                else 0
            ),
        },
    }
