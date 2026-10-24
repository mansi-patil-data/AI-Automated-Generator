"""Create downloadable Markdown business insight reports."""

from __future__ import annotations

from typing import Any


def generate_markdown_report(
    metrics: dict[str, Any],
    insights: dict[str, Any] | None = None,
    currency_symbol: str = "₹",
) -> str:
    """Render analyzed KPIs and optional AI recommendations as Markdown."""
    totals = metrics["totals"]
    availability = metrics.get("availability", {})
    amount = lambda key: (
        f"{currency_symbol}{totals[key]:,.2f}"
        if availability.get(key, True)
        else "Not available"
    )
    profit_margin = (
        f"{totals['profit_margin']:.2f}%"
        if availability.get("profit_margin", True)
        else "Not available"
    )
    average_order_value = (
        f"{currency_symbol}{totals['average_order_value']:,.2f}"
        if availability.get("orders", True)
        else "Not available"
    )
    order_count = (
        f"{totals['order_count']:,}"
        if availability.get("orders", True)
        else "Not available"
    )
    lines = [
        "# Business Insights Report",
        "",
        "## Executive KPIs",
        "",
        f"- **Revenue:** {amount('revenue')}",
        f"- **Expenses:** {amount('expenses')}",
        f"- **Profit:** {amount('profit')}",
        f"- **Profit margin:** {profit_margin}",
        f"- **Average order value:** {average_order_value}",
        f"- **Orders counted:** {order_count}",
        "",
        "## Performance",
        "",
    ]
    mom = totals.get("mom_growth")
    lines.append(
        f"- **Latest month-over-month revenue growth:** {mom:.2f}%"
        if mom is not None and availability.get("revenue", True)
        else "- **Latest month-over-month revenue growth:** Not available (revenue data or a non-zero prior month is required)."
    )
    for label, key in (
        ("Top region", "best_region"),
        ("Lowest region", "worst_region"),
        ("Top product", "best_product"),
        ("Lowest product", "worst_product"),
    ):
        item = metrics.get(key)
        lines.append(
            f"- **{label}:** {item['name']} ({currency_symbol}{item['revenue']:,.2f} revenue)"
            if item
            else f"- **{label}:** Not available"
        )

    trends = metrics.get("monthly_trends", [])
    if trends:
        columns = ["Month"]
        if availability.get("revenue", True):
            columns.append("Revenue")
        if availability.get("profit", True):
            columns.append("Profit")
        lines.extend(
            [
                "",
                "## Monthly Trends",
                "",
                "| " + " | ".join(columns) + " |",
                "|" + "|".join("---:" if column != "Month" else "---" for column in columns) + "|",
            ]
        )
        for row in trends:
            values = [row["month"]]
            if availability.get("revenue", True):
                values.append(f"{currency_symbol}{row['revenue']:,.2f}")
            if availability.get("profit", True):
                values.append(f"{currency_symbol}{row['profit']:,.2f}")
            lines.append("| " + " | ".join(values) + " |")

    if insights:
        source = insights.get("source", "AI")
        lines.extend(
            ["", f"## {source} Executive Summary", "", insights["executive_summary"], ""]
        )
        lines.extend(["## Recommendations", ""])
        lines.extend(f"- {recommendation}" for recommendation in insights["recommendations"])
        if insights.get("risks"):
            lines.extend(["", "## Risks to Monitor", ""])
            lines.extend(f"- {risk}" for risk in insights["risks"])

    lines.extend(
        [
            "",
            "---",
            f"Rows analyzed: {metrics.get('data_quality', {}).get('rows', 'unknown')}",
            "",
        ]
    )
    return "\n".join(lines)
