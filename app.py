"""Streamlit dashboard for automated business insights."""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from fpdf import FPDF
from google import genai
from google.genai.errors import APIError

from src.dataloader import DataValidationError, load_business_data
from src.dataanalyzer import analyze_data
from src.insightgenerator import (
    InsightGenerationError,
    generate_insights,
    generate_local_insights,
)
from src.reportgenerator import generate_markdown_report

voice_assistant = components.declare_component(
    "voice_assistant",
    path=str(Path(__file__).parent / "src" / "voice_component"),
)
currency_symbol = "₹"


def _detect_anomalies(metrics: dict[str, Any], threshold: float = 10.0) -> list[str]:
    """Find material monthly revenue declines and weakest category performers."""
    anomalies: list[str] = []
    trends = metrics.get("monthly_trends", [])
    for previous, current in zip(trends, trends[1:]):
        previous_revenue = previous["revenue"]
        if previous_revenue <= 0:
            continue
        change_pct = (current["revenue"] - previous_revenue) / previous_revenue * 100
        if abs(change_pct) >= threshold:
            direction = "dropped" if change_pct < 0 else "increased"
            anomalies.append(
                f"{current['month']} revenue {direction} by {abs(change_pct):.1f}% "
                f"from {previous['month']} "
                f"({currency_symbol}{previous_revenue:,.2f}) to "
                f"{currency_symbol}{current['revenue']:,.2f}."
            )

    for label, key in (
        ("region", "region_performance"),
        ("product", "product_performance"),
    ):
        performers = metrics.get(key, [])
        if len(performers) > 1:
            lowest = performers[-1]
            anomalies.append(
                f"Lowest-revenue {label}: {lowest['name']} "
                f"({currency_symbol}{lowest['revenue']:,.2f})."
            )
    return anomalies


def _answer_data_question(
    question: str,
    metrics: dict[str, Any],
    api_key: str,
    model: str = "gemini-2.5-flash",
    data: pd.DataFrame | None = None,
) -> str:
    """Use Gemini when available and fall back to deterministic Pandas answers."""
    if not api_key.strip():
        return (
            "Local analytics (using the current filtered dataset; Gemini is not configured): "
            + _answer_data_question_locally(question, metrics, data)
        )

    try:
        gemini = genai.Client(api_key=api_key)
        response = gemini.models.generate_content(
            model=model,
            contents=(
                "Answer the user's business question using only the supplied "
                "aggregated dashboard metrics. Do not invent values, infer missing "
                "rows, or claim causation. For comparisons, cite the values and "
                f"difference when available. Format financial values as {currency_symbol}. "
                "If the metrics do not contain enough "
                "detail, say what is missing.\n\n"
                f"Dashboard metrics (JSON): {json.dumps(metrics, allow_nan=False)}\n\n"
                f"Question: {question}"
            ),
            config={"temperature": 0.2},
        )
        answer = response.text.strip() if response.text else ""
        if answer:
            return "Gemini (based on aggregated dashboard metrics): " + answer
        fallback_reason = "Gemini returned an empty response"
    except APIError:
        fallback_reason = "Gemini is unavailable or rate-limited"
    except ValueError:
        fallback_reason = "Gemini returned an unusable response"
    except Exception:
        fallback_reason = "Gemini request failed"

    return (
        f"{fallback_reason}; switched to local analytics using the current filtered dataset. "
        + _answer_data_question_locally(question, metrics, data)
    )


def _answer_data_question_locally(
    question: str,
    metrics: dict[str, Any],
    data: pd.DataFrame | None = None,
) -> str:
    """Handle common dashboard questions with deterministic Pandas operations."""
    query = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
    totals = metrics["totals"]

    def currency(value: float) -> str:
        return f"{currency_symbol}{value:,.2f}"

    def metric_column() -> tuple[str, str]:
        if any(word in query for word in ("profit", "margin")):
            return "profit", "profit"
        if any(word in query for word in ("expense", "cost", "spend")):
            return "expenses", "expenses"
        return "revenue", "revenue"

    def group_result(
        frame: pd.DataFrame,
        dimension: str,
        value_column: str,
        direction: str,
    ) -> tuple[str, float] | None:
        if dimension not in frame or value_column not in frame:
            return None
        grouped = (
            frame.groupby(dimension, dropna=True)[value_column]
            .sum(min_count=1)
            .dropna()
        )
        if grouped.empty:
            return None
        name = grouped.idxmax() if direction == "highest" else grouped.idxmin()
        return str(name), float(grouped.loc[name])

    if data is not None and not data.empty:
        frame = data.copy()
        for column in ("revenue", "expenses", "profit"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if (
            ("profit" not in frame or not frame["profit"].notna().any())
            and {"revenue", "expenses"}.issubset(frame.columns)
        ):
            frame["profit"] = frame["revenue"] - frame["expenses"]

        value_column, value_label = metric_column()
        if value_column == "profit" and value_column not in frame:
            if {"revenue", "expenses"}.issubset(frame.columns):
                frame["profit"] = frame["revenue"] - frame["expenses"]
            else:
                return "Profit is unavailable because the dataset does not contain both profit or compatible revenue and expense data."

        is_highest = any(word in query for word in ("highest", "most", "top", "best", "maximum"))
        is_lowest = any(word in query for word in ("lowest", "least", "bottom", "worst", "minimum"))
        wants_compare = any(word in query for word in ("compare", "versus", " vs ", "difference between"))

        if any(
            word in query
            for word in ("growth", "mom", "month over month", "month on month")
        ):
            trends = metrics.get("monthly_trends", [])
            growth = metrics["totals"].get("mom_growth")
            if not metrics.get("availability", {}).get("revenue", True):
                return "Revenue growth is unavailable because revenue data was not detected."
            if len(trends) < 2 or growth is None:
                return "Month-over-month revenue growth is unavailable; two months and non-zero prior-month revenue are required."
            direction = "growth" if growth >= 0 else "decline"
            return (
                f"Revenue {direction} was {abs(growth):.1f}% from "
                f"{trends[-2]['month']} to {trends[-1]['month']}."
            )

        if "date" in frame and any(word in query for word in ("month", "monthly")):
            dated = frame.assign(_date=pd.to_datetime(frame["date"], errors="coerce")).dropna(
                subset=["_date"]
            )
            if not dated.empty and value_column in dated:
                dated["_month"] = dated["_date"].dt.to_period("M").astype(str)
                monthly = dated.groupby("_month")[value_column].sum(min_count=1).dropna()
                if not monthly.empty and (is_highest or is_lowest):
                    month = monthly.idxmax() if is_highest else monthly.idxmin()
                    value = float(monthly.loc[month])
                    return (
                        f"{month} had the {('highest' if is_highest else 'lowest')} "
                        f"{value_label}: {currency(value)}."
                    )

        if wants_compare:
            dimensions = []
            if "product" in frame:
                dimensions.append(("product", "product"))
            if "region" in frame:
                dimensions.append(("region", "region"))
            normalized_query = f" {query} "
            for dimension, label in dimensions:
                candidates = frame[dimension].dropna().astype(str).unique()
                matched = []
                normalized_query_words = [
                    word[:-3] + "y"
                    if word.endswith("ies")
                    else word[:-1]
                    if word.endswith("s") and not word.endswith("ss")
                    else word
                    for word in query.split()
                ]
                singular_query = f" {' '.join(normalized_query_words)} "
                for candidate in candidates:
                    normalized_name = re.sub(
                        r"[^a-z0-9]+", " ", candidate.casefold()
                    ).strip()
                    name_words = normalized_name.split()
                    if name_words:
                        last_word = name_words[-1]
                        name_words[-1] = (
                            last_word[:-3] + "y"
                            if last_word.endswith("ies")
                            else last_word[:-1]
                            if last_word.endswith("s") and not last_word.endswith("ss")
                            else last_word
                        )
                    singular_name = " ".join(name_words)
                    if (
                        f" {normalized_name} " in normalized_query
                        or (singular_name and f" {singular_name} " in singular_query)
                    ):
                        matched.append(candidate)
                if len(matched) >= 2 and value_column in frame:
                    comparison = (
                        frame[frame[dimension].astype(str).isin(matched)]
                        .groupby(dimension)[value_column]
                        .sum(min_count=1)
                        .dropna()
                    )
                    if len(comparison) >= 2:
                        items = sorted(
                            ((str(name), float(value)) for name, value in comparison.items()),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                        first, second = items[:2]
                        difference = abs(first[1] - second[1])
                        return (
                            f"{label.title()} comparison by {value_label}: "
                            f"{first[0]} {currency(first[1])}; {second[0]} "
                            f"{currency(second[1])}. Difference: {currency(difference)}."
                        )

        if (is_highest or is_lowest) and value_column in frame:
            direction = "highest" if is_highest else "lowest"
            dimension = (
                "product" if any(word in query for word in ("product", "item", "category"))
                else "region" if "region" in query
                else None
            )
            if dimension:
                result = group_result(frame, dimension, value_column, direction)
                if result:
                    return (
                        f"{result[0]} had the {direction} {value_label}: "
                        f"{currency(result[1])}."
                    )

    if any(word in query for word in ("total", "overall", "sum")):
        available = metrics.get("availability", {})
        requested_keys = []
        if any(word in query for word in ("revenue", "sales")):
            requested_keys.append(("revenue", "total revenue"))
        if any(word in query for word in ("expense", "cost", "spend")):
            requested_keys.append(("expenses", "total expenses"))
        if "profit" in query:
            requested_keys.append(("profit", "total profit"))
        if not requested_keys:
            requested_keys = [
                ("revenue", "total revenue"),
                ("expenses", "total expenses"),
                ("profit", "total profit"),
            ]
        available_totals = [
            f"{label} is {currency(float(totals[key]))}"
            for key, label in requested_keys
            if available.get(key, True)
        ]
        if not available_totals:
            return (
                "The requested metric is unavailable because the dataset does not "
                "contain compatible usable values."
            )
        unavailable = [
            label
            for key, label in requested_keys
            if not available.get(key, True)
        ]
        response = "; ".join(available_totals) + "."
        if unavailable:
            response += " Not available from this dataset: " + ", ".join(unavailable) + "."
        return response
    if "average" in query and any(word in query for word in ("order", "value", "sale")):
        if not metrics.get("availability", {}).get("orders", True):
            return "Average order value is unavailable because there are no usable revenue values."
        return f"Average order value is {currency(float(totals['average_order_value']))}."
    if "margin" in query:
        if not metrics.get("availability", {}).get("profit_margin", True):
            return "Profit margin is unavailable because valid revenue and profit data are required."
        return f"Profit margin is {float(totals['profit_margin']):.1f}%."
    if any(word in query for word in ("expense", "cost", "spend")):
        if not metrics.get("availability", {}).get("expenses", True):
            return "Expenses are unavailable because no usable expense values were detected."
        return f"Total expenses are {currency(float(totals['expenses']))}."
    if "profit" in query:
        if not metrics.get("availability", {}).get("profit", True):
            return "Profit is unavailable because the dataset does not contain profit or compatible revenue and expense data."
        return f"Total profit is {currency(float(totals['profit']))}."
    if any(word in query for word in ("revenue", "sales", "sale")):
        if not metrics.get("availability", {}).get("revenue", True):
            return "Revenue is unavailable because no usable revenue values were detected."
        return f"Total revenue is {currency(float(totals['revenue']))}."

    best_product = metrics.get("best_product")
    best_region = metrics.get("best_region")
    available = metrics.get("availability", {})
    summary_parts = []
    if available.get("revenue", True):
        summary_parts.append(f"Revenue: {currency(float(totals['revenue']))}")
    if available.get("profit", True):
        summary_parts.append(f"profit: {currency(float(totals['profit']))}")
    if best_product:
        summary_parts.append(
            f"top product: {best_product['name']} "
            f"({currency(float(best_product['revenue']))} revenue)"
        )
    if best_region:
        summary_parts.append(
            f"top region: {best_region['name']} "
            f"({currency(float(best_region['revenue']))} revenue)"
        )
    return (
        "Local analytics could not identify a specific calculation for that question. "
        "Try asking for totals, profit margin, a highest/lowest month, product, or region. "
        "Current data summary: " + "; ".join(summary_parts) + "."
    )


def _blend_hex(color: str, target: str, amount: float) -> str:
    """Blend a six-digit hex color toward another color."""
    source_rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    target_rgb = tuple(int(target[index:index + 2], 16) for index in (1, 3, 5))
    blended = tuple(
        round(source * (1 - amount) + destination * amount)
        for source, destination in zip(source_rgb, target_rgb)
    )
    return "#" + "".join(f"{channel:02X}" for channel in blended)


def _hex_to_rgba(color: str, opacity: float) -> str:
    """Convert a six-digit hex color to CSS rgba notation."""
    red, green, blue = (
        int(color[index:index + 2], 16) for index in (1, 3, 5)
    )
    return f"rgba({red}, {green}, {blue}, {opacity})"


def _contrast_text(color: str) -> str:
    """Choose a readable foreground for the selected background color."""
    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return "#0F172A" if luminance > 0.42 else "#FFFFFF"


def _generate_pdf_report(
    metrics: dict[str, Any],
    insights: dict[str, Any] | None = None,
    anomalies: list[str] | None = None,
) -> bytes:
    """Build a compact executive PDF report using fpdf2."""
    totals = metrics["totals"]
    availability = metrics.get("availability", {})
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_title("Business Insights Executive Report")
    unicode_font_path = next(
        (
            candidate
            for candidate in (
                Path(r"C:\Windows\Fonts\arial.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            )
            if candidate.is_file()
        ),
        None,
    )
    if unicode_font_path is not None:
        pdf.add_font("DashboardUnicode", fname=str(unicode_font_path))
    report_font = "DashboardUnicode" if unicode_font_path is not None else "Helvetica"
    currency_prefix = (
        currency_symbol
        if unicode_font_path is not None or currency_symbol.isascii()
        else "INR "
    )
    pdf.set_font("Helvetica", "B", 19)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(
        pdf.epw,
        12,
        "Business Insights Executive Report",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(3)

    def add_section(title: str) -> None:
        pdf.ln(3)
        pdf.set_fill_color(16, 185, 129)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(pdf.epw, 9, title, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(31, 41, 55)

    def add_bullet(text: str) -> None:
        pdf.set_font(report_font, size=10)
        pdf_safe_text = (
            text
            if unicode_font_path is not None or currency_symbol.isascii()
            else text.replace(currency_symbol, "INR ")
        )
        if unicode_font_path is None:
            pdf_safe_text = pdf_safe_text.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(pdf.epw, 6, f"- {pdf_safe_text}")

    add_section("Executive KPIs")
    kpi_rows = [
        (
            "Revenue",
            f"{currency_prefix}{totals['revenue']:,.2f}"
            if availability.get("revenue", True)
            else "Not available",
        ),
        (
            "Expenses",
            f"{currency_prefix}{totals['expenses']:,.2f}"
            if availability.get("expenses", True)
            else "Not available",
        ),
        (
            "Profit",
            f"{currency_prefix}{totals['profit']:,.2f}"
            if availability.get("profit", True)
            else "Not available",
        ),
        (
            "Profit margin",
            f"{totals['profit_margin']:.2f}%"
            if availability.get("profit_margin", True)
            else "Not available",
        ),
        (
            "Average order value",
            f"{currency_prefix}{totals['average_order_value']:,.2f}"
            if availability.get("orders", True)
            else "Not available",
        ),
    ]
    pdf.set_font(report_font, size=10)
    for label, value in kpi_rows:
        pdf.cell(95, 7, label, border=1)
        pdf.cell(
            pdf.epw - 95,
            7,
            value,
            border=1,
            new_x="LMARGIN",
            new_y="NEXT",
        )

    add_section("Performance highlights")
    for label, key in (
        ("Top region", "best_region"),
        ("Lowest region", "worst_region"),
        ("Top product", "best_product"),
        ("Lowest product", "worst_product"),
    ):
        performer = metrics.get(key)
        value = (
            f"{performer['name']} ({currency_prefix}{performer['revenue']:,.2f} revenue)"
            if performer
            else "Not available"
        )
        add_bullet(f"{label}: {value}")

    monthly_trends = metrics.get("monthly_trends", [])
    if monthly_trends:
        add_section("Monthly trends")
        for row in monthly_trends:
            values = []
            if availability.get("revenue", True):
                values.append(f"revenue {currency_prefix}{row['revenue']:,.2f}")
            if availability.get("profit", True):
                values.append(f"profit {currency_prefix}{row['profit']:,.2f}")
            add_bullet(f"{row['month']}: " + "; ".join(values))

    if insights:
        add_section(f"{insights.get('source', 'AI')} executive summary")
        add_bullet(str(insights.get("executive_summary", "")))
        add_section("Recommendations")
        for item in insights.get("recommendations", []):
            add_bullet(str(item))
        if insights.get("risks"):
            add_section("Risks to monitor")
            for item in insights["risks"]:
                add_bullet(str(item))

    if anomalies:
        add_section("Automated risk alerts")
        for item in anomalies:
            add_bullet(item)

    return bytes(pdf.output())


def _generate_csv_report(
    metrics: dict[str, Any],
    insights: dict[str, Any] | None = None,
) -> str:
    """Render KPIs and supporting dashboard data as a sectioned CSV."""
    totals = metrics["totals"]
    availability = metrics.get("availability", {})
    rows: list[dict[str, str | int | float]] = [
        {"Section": "Settings", "Name": "Display currency", "Value": currency_symbol}
    ]
    for key, label in (
        ("revenue", "Revenue"),
        ("expenses", "Expenses"),
        ("profit", "Profit"),
        ("profit_margin", "Profit margin (%)"),
        ("average_order_value", "Average order value"),
    ):
        is_available = availability.get(
            "orders" if key == "average_order_value" else key,
            True,
        )
        if is_available:
            rows.append({"Section": "KPI", "Name": label, "Value": totals[key]})
    for section, key in (
        ("Region", "region_performance"),
        ("Product", "product_performance"),
    ):
        rows.extend(
            {"Section": section, "Name": item["name"], "Value": item["revenue"]}
            for item in metrics.get(key, [])
        )
    for trend in metrics.get("monthly_trends", []):
        values = []
        if availability.get("revenue", True):
            values.append(f"Revenue: {trend['revenue']:.2f}")
        if availability.get("profit", True):
            values.append(f"Profit: {trend['profit']:.2f}")
        if values:
            rows.append(
                {
                    "Section": "Monthly trend",
                    "Name": trend["month"],
                    "Value": trend["revenue"] if availability.get("revenue", True) else trend["profit"],
                    "Details": "; ".join(values),
                }
            )
    if insights:
        rows.append(
            {
                "Section": f"{insights.get('source', 'AI')} summary",
                "Name": "Executive summary",
                "Value": insights["executive_summary"],
            }
        )
        rows.extend(
            {"Section": "AI recommendation", "Name": str(index + 1), "Value": item}
            for index, item in enumerate(insights.get("recommendations", []))
        )
        rows.extend(
            {"Section": "AI risk", "Name": str(index + 1), "Value": item}
            for index, item in enumerate(insights.get("risks", []))
        )
    return pd.DataFrame(rows).to_csv(index=False)


load_dotenv()
st.set_page_config(page_title="Automated Insight Generator", page_icon="📊", layout="wide")

currency_options = {
    "INR (₹)": "₹",
    "USD ($)": "$",
    "EUR (€)": "€",
    "GBP (£)": "£",
}
currency_symbol = currency_options[
    st.sidebar.selectbox(
        "Display currency (set to match your data)",
        options=list(currency_options),
        key="display_currency",
    )
]
st.sidebar.header("🎨 Dashboard Theme Customizer")
accent_color = st.sidebar.color_picker(
    "Primary accent color",
    "#10B981",
    key="dashboard_accent_color",
)
card_background = st.sidebar.color_picker(
    "Card background color",
    "#FFFFFF",
    key="dashboard_card_background",
)
dark_mode = st.sidebar.toggle("🌙 Dark mode", value=False, key="dashboard_dark_mode")
image_export_format = st.sidebar.selectbox(
    "Chart image export format",
    ("PNG", "SVG"),
    help="Choose the format used by the Plotly chart modebar's camera button.",
    key="chart_image_export_format",
)
accent_light = _blend_hex(accent_color, "#FFFFFF", 0.72)
accent_lighter = _blend_hex(accent_color, "#FFFFFF", 0.88)
accent_dark = _blend_hex(accent_color, "#0F172A", 0.25)
effective_card_background = (
    _blend_hex(card_background, "#0B1220", 0.82) if dark_mode else card_background
)
card_background_rgba = _hex_to_rgba(effective_card_background, 0.88 if dark_mode else 0.82)
card_text = _contrast_text(effective_card_background)
app_background = "#0B1220" if dark_mode else "#F8FAFC"
surface_background = "#111827" if dark_mode else "#FFFFFF"
muted_text = "#94A3B8" if dark_mode else "#475569"
grid_color = "#334155" if dark_mode else "#E2E8F0"
text_color = "#F1F5F9" if dark_mode else "#0F172A"
chart_config = {
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["toImage"],
    "toImageButtonOptions": {
        "format": image_export_format.lower(),
        "filename": "automated_insight_chart",
        "height": 700,
        "width": 1200,
        "scale": 2,
    },
}
donut_colors = [
    accent_color,
    accent_light,
    _blend_hex(accent_color, "#6366F1", 0.42),
    accent_lighter,
    _blend_hex(accent_color, "#0F172A", 0.18),
    _blend_hex(accent_color, "#A5B4FC", 0.55),
]

st.markdown(
    """
    <style>
    .stApp {
        background: #F8FAFC;
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }

    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.04em !important;
    }

    h1 {
        font-size: 4rem !important;
        margin-bottom: 0.35rem !important;
    }

    h2, h3 {
        margin-top: 1.5rem !important;
    }

    .stCaption {
        color: #475569 !important;
        font-size: 1.1rem !important;
        margin-bottom: 1.5rem !important;
    }

    [data-testid="stFileUploader"] > section {
        background: #FFFFFF !important;
        border: 1.5px dashed #CBD5E1 !important;
        border-radius: 16px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
    }

    [data-testid="stFileUploader"] > section:hover {
        border-color: #10B981 !important;
        box-shadow: 0 12px 24px rgba(16, 185, 129, 0.12);
        transform: translateY(-1px);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF !important;
        border: 0 !important;
        min-height: 100px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.72rem 1.2rem !important;
        font-weight: 700 !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #10B981 0%, #0F172A 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(16, 185, 129, 0.18);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    }

    [data-testid="stMetric"],
    div[data-testid="metric-container"] {
        background: #FFFFFF !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        padding: 16px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
        font-size: 0.875rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stExpander"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    .stPlotlyChart {
        border-radius: 14px;
        overflow: hidden;
    }

    .stAlert {
        border-radius: 12px !important;
        border: 1px solid #D1FAE5 !important;
        background: #F0FDF4 !important;
    }

    .success-badge {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-top: 0.35rem;
    }

    .stSidebar {
        background: #F8FAFC;
    }

    .executive-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #E5E7EB;
        border-left: 4px solid #10B981;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        color: #334155;
        line-height: 1.65;
        padding: 16px 20px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .executive-card:hover {
        border-color: #10B981;
        box-shadow: 0 0 0 1px #10B981, 0 12px 28px rgba(16, 185, 129, 0.24);
    }

    .executive-card h3 {
        color: #0F172A;
        font-size: 1.1rem;
        letter-spacing: 0 !important;
        margin: 0 0 0.6rem;
    }

    .executive-card ul {
        margin: 0.4rem 0 0;
        padding-left: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
theme_css = """
<style>
[data-testid="stMetric"],
div[data-testid="metric-container"] {
    background: __CARD_BACKGROUND__ !important;
    color: __CARD_TEXT__ !important;
    border: 1px solid __ACCENT_LIGHT__ !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
[data-testid="stMetric"]:hover,
div[data-testid="metric-container"]:hover {
    border-color: __ACCENT__ !important;
    box-shadow: 0 0 0 1px __ACCENT__, 0 12px 28px __ACCENT_GLOW__ !important;
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: __CARD_TEXT__ !important;
}
.stApp,
[data-testid="stAppViewContainer"] > .main {
    background: __APP_BACKGROUND__ !important;
    color: __TEXT_COLOR__ !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: __APP_BACKGROUND__ !important;
    color: __TEXT_COLOR__ !important;
}
h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
    color: __TEXT_COLOR__ !important;
}
.stCaption,
[data-testid="stCaptionContainer"] {
    color: __MUTED_TEXT__ !important;
}
[data-testid="stFileUploader"] > section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stExpander"],
[data-testid="stDataFrame"] {
    background: __SURFACE_BACKGROUND__ !important;
    color: __TEXT_COLOR__ !important;
}
[data-testid="stAlert"] {
    background: __SURFACE_BACKGROUND__ !important;
    color: __TEXT_COLOR__ !important;
}
.stPlotlyChart {
    background: __CARD_BACKGROUND__;
    border: 1px solid __ACCENT_LIGHT__;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
}
.stPlotlyChart:hover {
    border-color: __ACCENT__;
    box-shadow: 0 0 0 1px __ACCENT__, 0 12px 28px __ACCENT_GLOW__;
    transform: translateY(-1px);
}
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, __ACCENT__ 0%, __ACCENT_DARK__ 100%) !important;
    border: 1px solid __ACCENT_LIGHT__ !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    box-shadow: 0 0 0 1px __ACCENT__, 0 8px 22px __ACCENT_GLOW__ !important;
}
.success-badge {
    background: __ACCENT_LIGHT__;
    color: __ACCENT_DARK__;
}
.executive-card {
    background: __CARD_BACKGROUND__;
    color: __CARD_TEXT__;
    border-color: __ACCENT_LIGHT__;
    border-left-color: __ACCENT__;
}
.executive-card:hover {
    border-color: __ACCENT__;
    box-shadow: 0 0 0 1px __ACCENT__, 0 12px 28px __ACCENT_GLOW__;
}
.executive-card h3 {
    color: __CARD_TEXT__;
}
</style>
"""
for placeholder, value in (
    ("__ACCENT_GLOW__", _hex_to_rgba(accent_color, 0.24)),
    ("__ACCENT_LIGHT__", accent_light),
    ("__ACCENT_DARK__", accent_dark),
    ("__CARD_BACKGROUND__", card_background_rgba),
    ("__CARD_TEXT__", card_text),
    ("__ACCENT__", accent_color),
    ("__APP_BACKGROUND__", app_background),
    ("__SURFACE_BACKGROUND__", surface_background),
    ("__MUTED_TEXT__", muted_text),
    ("__TEXT_COLOR__", text_color),
):
    theme_css = theme_css.replace(placeholder, value)
st.markdown(theme_css, unsafe_allow_html=True)

header_slot = st.empty()
with header_slot.container():
    title_column, _ = st.columns([3, 1])
    with title_column:
        st.title("📊 Automated Insight Generator")
        st.caption("Turn business exports into clear KPIs, trends, and executive recommendations.")

uploaded_file = st.file_uploader(
    "Upload a business data export",
    type=("csv", "xlsx", "xls"),
    help="Supported formats: CSV and Excel workbooks.",
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to view the dashboard.")
    sample_path = Path(__file__).parent / "sampledata" / "salesdata.csv"
    if sample_path.exists() and st.button("Load the included sample dataset"):
        st.session_state["sample_data"] = sample_path.read_bytes()

file_bytes: bytes | None = None
file_name: str | None = None
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
elif st.session_state.get("sample_data") is not None:
    file_bytes = st.session_state["sample_data"]
    file_name = "salesdata.csv"

if file_bytes is not None and file_name is not None:
    try:
        data = load_business_data(file_bytes, file_name)
    except (DataValidationError, ValueError, OSError) as exc:
        st.error(f"Could not load this file: {exc}")
        st.stop()

    date_range = "All dates"
    filtered_data = data
    if "date" in data.columns:
        parsed_dates = pd.to_datetime(data["date"], errors="coerce")
        usable_dates = parsed_dates.dropna()
        if not usable_dates.empty:
            latest_date = usable_dates.max()
            latest_year = latest_date.year
            date_range_options = [
                "All dates",
                "Last 6 Months",
                f"Q1 {latest_year}",
                f"YTD {latest_year}",
                f"Jan-Jun {latest_year}",
            ]
            with header_slot.container():
                title_column, filter_column = st.columns([3, 1])
                with title_column:
                    st.title("📊 Automated Insight Generator")
                    st.caption(
                        "Turn business exports into clear KPIs, trends, and executive recommendations."
                    )
                with filter_column:
                    st.selectbox(
                        "Date range",
                        date_range_options,
                        key=f"date_range_filter_{latest_year}",
                    )
            date_range = st.session_state[f"date_range_filter_{latest_year}"]

            if date_range == "Last 6 Months":
                start_date = latest_date - pd.DateOffset(months=6)
                date_mask = parsed_dates.between(start_date, latest_date)
            elif date_range == f"Q1 {latest_year}":
                date_mask = parsed_dates.between(
                    pd.Timestamp(year=latest_year, month=1, day=1),
                    pd.Timestamp(year=latest_year, month=4, day=1),
                    inclusive="left",
                )
            elif date_range == f"YTD {latest_year}":
                date_mask = parsed_dates.between(
                    pd.Timestamp(year=latest_year, month=1, day=1),
                    latest_date,
                )
            elif date_range == f"Jan-Jun {latest_year}":
                date_mask = parsed_dates.between(
                    pd.Timestamp(year=latest_year, month=1, day=1),
                    pd.Timestamp(year=latest_year, month=7, day=1),
                    inclusive="left",
                )
            else:
                date_mask = pd.Series(True, index=data.index)
            filtered_data = data.loc[date_mask].copy()
        else:
            st.sidebar.caption("No usable dates found; showing all rows.")
    else:
        st.sidebar.caption("No date column found; showing all rows.")

    if filtered_data.empty:
        st.warning(f"No rows match the selected date range ({date_range}). Choose another range.")
        st.stop()

    dataset_key = abs(hash((file_name, hash(file_bytes), date_range)))
    st.sidebar.header("Dashboard filters")
    selected_filters: dict[str, list[str]] = {}
    if "date" in filtered_data.columns:
        month_values = pd.to_datetime(
            filtered_data["date"],
            errors="coerce",
        ).dt.to_period("M").astype(str)
        month_options = sorted(
            month for month in month_values.unique().tolist() if month != "NaT"
        )
        selected_months = st.sidebar.multiselect(
            "Months",
            options=month_options,
            default=month_options,
            key=f"dashboard_filter_month_{dataset_key}",
        )
        selected_filters["month"] = selected_months
        if len(selected_months) < len(month_options):
            filtered_data = filtered_data.loc[
                month_values.isin(selected_months)
            ].copy()

    for dimension, label in (
        ("region", "Regions"),
        ("product", "Products"),
        ("category", "Categories"),
    ):
        if dimension not in filtered_data.columns:
            continue
        options = sorted(
            filtered_data[dimension].dropna().astype(str).unique().tolist()
        )
        selected = st.sidebar.multiselect(
            label,
            options=options,
            default=options,
            key=f"dashboard_filter_{dimension}_{dataset_key}",
        )
        selected_filters[dimension] = selected
        if len(selected) < len(options):
            filtered_data = filtered_data[
                filtered_data[dimension].astype(str).isin(selected)
            ].copy()

    if filtered_data.empty:
        st.warning("No rows match the selected filters. Select at least one available value.")
        st.stop()

    comparison_key = abs(hash((dataset_key, tuple(
        (dimension, tuple(values))
        for dimension, values in selected_filters.items()
    ))))
    region_options = (
        sorted(filtered_data["region"].dropna().astype(str).unique().tolist())
        if "region" in filtered_data.columns
        else []
    )
    product_options = (
        sorted(filtered_data["product"].dropna().astype(str).unique().tolist())
        if "product" in filtered_data.columns
        else []
    )
    st.sidebar.header("Compare regions and products")
    selected_regions = st.sidebar.multiselect(
        "Regions to compare",
        options=region_options,
        key=f"compare_regions_{comparison_key}",
        help="Select regions to compare their revenue side-by-side.",
    )
    selected_products = st.sidebar.multiselect(
        "Products to compare",
        options=product_options,
        key=f"compare_products_{comparison_key}",
        help="Select products to compare their revenue side-by-side.",
    )

    try:
        metrics = analyze_data(filtered_data)
    except ValueError as exc:
        st.error(f"Could not analyze this file: {exc}")
        st.stop()

    insight_context = (
        file_name,
        date_range,
        currency_symbol,
        tuple((dimension, tuple(values)) for dimension, values in selected_filters.items()),
        len(file_bytes),
        hash(file_bytes),
    )
    if st.session_state.get("_insights_context") != insight_context:
        st.session_state.pop("ai_insights", None)
        st.session_state.pop("ai_insights_error", None)
        st.session_state.pop("data_chat_history", None)
        st.session_state["_insights_context"] = insight_context

    anomalies = _detect_anomalies(metrics)
    availability = metrics.get("availability", {})
    top_product = metrics.get("best_product")
    st.sidebar.header("🔮 What-If Scenario Simulator")
    projected_sales_growth = st.sidebar.slider(
        "Projected top-product sales growth (%)",
        min_value=0,
        max_value=100,
        value=10,
        step=5,
        key=f"scenario_growth_{abs(hash(insight_context))}",
    )
    projected_cost_reduction = st.sidebar.slider(
        "Projected cost reduction (%)",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        key=f"scenario_cost_{abs(hash(insight_context))}",
    )
    revenue_increase = (
        top_product["revenue"] * projected_sales_growth / 100 if top_product else 0.0
    )
    cost_savings = (
        metrics["totals"]["expenses"] * projected_cost_reduction / 100
        if availability.get("expenses", True)
        else 0.0
    )
    simulated_revenue = metrics["totals"]["revenue"] + revenue_increase
    simulated_expenses = metrics["totals"]["expenses"] - cost_savings
    simulated_profit = metrics["totals"]["profit"] + revenue_increase + cost_savings
    simulated_margin = (
        simulated_profit / simulated_revenue * 100
        if availability.get("revenue", True)
        and availability.get("profit", True)
        and simulated_revenue
        else None
    )
    st.sidebar.caption(
        f"Growth is applied to {top_product['name']}."
        if top_product
        else "No product data available; revenue remains unchanged."
    )
    st.sidebar.metric(
        "Simulated revenue",
        f"{currency_symbol}{simulated_revenue:,.2f}"
        if availability.get("revenue", True)
        else "Not available",
    )
    st.sidebar.metric(
        "Simulated expenses",
        f"{currency_symbol}{simulated_expenses:,.2f}"
        if availability.get("expenses", True)
        else "Not available",
    )
    st.sidebar.metric(
        "Simulated profit margin",
        f"{simulated_margin:.1f}%" if simulated_margin is not None else "Not available",
    )

    st.subheader("Dataset overview")
    st.write(
        f"{len(filtered_data):,} of {len(data):,} validated rows · "
        f"{len(data.columns):,} columns · {date_range}"
    )
    with st.expander("Data quality checks"):
        missing_counts = filtered_data.isna().sum()
        st.write(
            f"Duplicate rows removed during loading: "
            f"{data.attrs.get('duplicates_removed', 0):,} "
            f"of {data.attrs.get('source_rows', len(data)):,} source rows."
        )
        missing_summary = pd.DataFrame(
            {
                "Column": missing_counts.index,
                "Missing values": missing_counts.values,
                "Missing (%)": (
                    missing_counts.values / len(filtered_data) * 100
                ).round(1),
            }
        )
        st.dataframe(missing_summary, hide_index=True, use_container_width=True)
    with st.expander("Preview normalized data"):
        st.dataframe(filtered_data.head(20), use_container_width=True)

    totals = metrics["totals"]
    cards = st.columns(5)
    cards[0].metric(
        "💰 Revenue",
        f"{currency_symbol}{totals['revenue']:,.2f}" if availability.get("revenue", True) else "Not available",
    )
    cards[1].metric(
        "🧾 Expenses",
        f"{currency_symbol}{totals['expenses']:,.2f}" if availability.get("expenses", True) else "Not available",
    )
    cards[2].metric(
        "📈 Profit",
        f"{currency_symbol}{totals['profit']:,.2f}" if availability.get("profit", True) else "Not available",
    )
    cards[3].metric(
        "📊 Profit margin",
        f"{totals['profit_margin']:.1f}%"
        if availability.get("profit_margin", True)
        else "Not available",
    )
    cards[4].metric(
        "🛒 Orders counted",
        f"{totals['order_count']:,}"
        if availability.get("orders", True)
        else "Not available",
        help="Rows with usable revenue are counted as orders; no order ID is present.",
    )
    if availability.get("orders", True):
        st.caption(
            f"Average order value: {currency_symbol}"
            f"{totals['average_order_value']:,.2f}"
        )

    if totals.get("mom_growth") is not None and totals["mom_growth"] > 0:
        st.markdown(
            f'<div class="success-badge">+{totals["mom_growth"]:.1f}% MoM</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)
    monthly = metrics["monthly_trends"]
    with left:
        st.subheader("Monthly performance")
        monthly_values = [
            value
            for value in ("revenue", "profit")
            if availability.get(value, True)
        ]
        if monthly and monthly_values:
            monthly_frame = pd.DataFrame(monthly)
            fig = px.line(
                monthly_frame,
                x="month",
                y=monthly_values,
                markers=True,
                title="Revenue and profit by month",
                labels={"value": "Amount", "month": "Month", "variable": "Metric"},
                color_discrete_sequence=[
                    accent_color,
                    _blend_hex(accent_color, "#6366F1", 0.45),
                ],
            )
            fig.update_traces(mode="lines+markers", marker=dict(size=7), line=dict(width=3))
            fig.update_layout(
                paper_bgcolor=card_background_rgba,
                plot_bgcolor=card_background_rgba,
                title_font={"size": 18, "color": card_text},
                legend_title_text="Metric",
                xaxis=dict(
                    gridcolor=grid_color,
                    tickfont={"color": muted_text},
                ),
                yaxis=dict(
                    gridcolor=grid_color,
                    tickfont={"color": muted_text},
                    tickprefix=currency_symbol,
                ),
                font={"color": card_text},
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True, config=chart_config)
            if totals["mom_growth"] is not None:
                badge_text = f"+{totals['mom_growth']:.1f}% MoM" if totals['mom_growth'] > 0 else f"{totals['mom_growth']:.1f}% MoM"
                if totals['mom_growth'] > 0:
                    st.markdown(f'<div class="success-badge">{badge_text}</div>', unsafe_allow_html=True)
                else:
                    st.caption(f"Latest month-over-month revenue growth: {totals['mom_growth']:.1f}%")
        else:
            st.info("Add a usable date column to see monthly trends.")

    with right:
        st.subheader("Revenue by region")
        regions = metrics["region_performance"]
        if regions:
            region_frame = pd.DataFrame(regions)
            # Force correct column name for region
            if region_frame is not None and not region_frame.empty:
                if "name" in region_frame.columns and "region" not in region_frame.columns:
                    region_frame = region_frame.rename(columns={"name": "region"})

            total_region_revenue = region_frame["revenue"].sum()
            region_frame["revenue_share"] = (
                region_frame["revenue"] / total_region_revenue * 100
                if total_region_revenue
                else 0
            )
            fig = px.bar(
                region_frame,
                x="region",
                y="revenue",
                color="revenue",
                color_continuous_scale=[
                    [0, accent_lighter],
                    [0.5, accent_light],
                    [1, accent_color],
                ],
                custom_data=["revenue_share"],
                labels={"region": "Region", "revenue": "Revenue"},
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"Revenue: {currency_symbol}%{{y:,.2f}}<br>"
                    "Regional share: %{customdata[0]:.1f}%"
                    "<extra></extra>"
                ),
                marker_line_color=text_color,
                marker_line_width=0,
                width=0.7,
            )
            fig.update_layout(
                paper_bgcolor=card_background_rgba,
                plot_bgcolor=card_background_rgba,
                title_font={"size": 18, "color": card_text},
                xaxis=dict(
                    gridcolor=grid_color,
                    tickfont={"color": muted_text},
                ),
                yaxis=dict(
                    gridcolor=grid_color,
                    tickfont={"color": muted_text},
                    tickprefix=currency_symbol,
                ),
                coloraxis_showscale=False,
                font={"color": card_text},
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True, config=chart_config)
        else:
            st.info("Add a region column to compare regions.")

    st.subheader("Region and product comparison")
    comparison_region_column, comparison_product_column = st.columns(2)
    with comparison_region_column:
        if selected_regions:
            region_comparison = pd.DataFrame(metrics["region_performance"])
            region_comparison = region_comparison[
                region_comparison["name"].isin(selected_regions)
            ]
            compare_fig = px.bar(
                region_comparison,
                x="name",
                y="revenue",
                color="name",
                color_discrete_sequence=[
                    accent_color,
                    accent_light,
                    _blend_hex(accent_color, "#6366F1", 0.45),
                    accent_lighter,
                ],
                labels={"name": "Region", "revenue": "Revenue"},
                title="Selected regions",
            )
            compare_fig.update_traces(
                hovertemplate=f"<b>%{{x}}</b><br>Revenue: {currency_symbol}%{{y:,.2f}}<extra></extra>"
            )
            compare_fig.update_layout(
                paper_bgcolor=card_background_rgba,
                plot_bgcolor=card_background_rgba,
                font={"color": card_text},
                xaxis={"title": "Region", "tickfont": {"color": muted_text}},
                yaxis={
                    "title": "Revenue",
                    "gridcolor": grid_color,
                    "tickfont": {"color": muted_text},
                    "tickprefix": currency_symbol,
                },
                showlegend=False,
                margin=dict(l=20, r=20, t=45, b=20),
            )
            st.plotly_chart(compare_fig, use_container_width=True, config=chart_config)
        else:
            st.info("Select two or more regions in the sidebar to compare their revenue.")

    with comparison_product_column:
        if selected_products:
            product_comparison = pd.DataFrame(metrics["product_performance"])
            product_comparison = product_comparison[
                product_comparison["name"].isin(selected_products)
            ]
            compare_fig = px.bar(
                product_comparison,
                x="name",
                y="revenue",
                color="name",
                color_discrete_sequence=donut_colors,
                labels={"name": "Product", "revenue": "Revenue"},
                title="Selected products",
            )
            compare_fig.update_traces(
                hovertemplate=f"<b>%{{x}}</b><br>Revenue: {currency_symbol}%{{y:,.2f}}<extra></extra>"
            )
            compare_fig.update_layout(
                paper_bgcolor=card_background_rgba,
                plot_bgcolor=card_background_rgba,
                font={"color": card_text},
                xaxis={"title": "Product", "tickfont": {"color": muted_text}},
                yaxis={
                    "title": "Revenue",
                    "gridcolor": grid_color,
                    "tickfont": {"color": muted_text},
                    "tickprefix": currency_symbol,
                },
                showlegend=False,
                margin=dict(l=20, r=20, t=45, b=20),
            )
            st.plotly_chart(compare_fig, use_container_width=True, config=chart_config)
        else:
            st.info("Select two or more products in the sidebar to compare their revenue.")

    categories = metrics["category_performance"]
    if categories:
        st.subheader("Revenue by category")
        category_frame = pd.DataFrame(categories)
        category_fig = px.bar(
            category_frame,
            x="name",
            y="revenue",
            color="name",
            color_discrete_sequence=donut_colors,
            labels={"name": "Category", "revenue": "Revenue"},
        )
        category_fig.update_traces(
            hovertemplate=f"<b>%{{x}}</b><br>Revenue: {currency_symbol}%{{y:,.2f}}<extra></extra>"
        )
        category_fig.update_layout(
            paper_bgcolor=card_background_rgba,
            plot_bgcolor=card_background_rgba,
            font={"color": card_text},
            showlegend=False,
            xaxis={"title": "Category", "tickfont": {"color": muted_text}},
            yaxis={
                "title": "Revenue",
                "gridcolor": grid_color,
                "tickfont": {"color": muted_text},
                "tickprefix": currency_symbol,
            },
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(category_fig, use_container_width=True, config=chart_config)

    lower_left, lower_right = st.columns(2)
    products = metrics["product_performance"]
    with lower_left:
        st.subheader("Product revenue share")
        if products:
            product_frame = pd.DataFrame(products)
            product_revenue = product_frame["revenue"].sum()
            donut = px.pie(
                product_frame,
                names="name",
                values="revenue",
                hole=0.62,
                color_discrete_sequence=donut_colors,
                labels={"name": "Product", "revenue": "Revenue"},
            )
            donut.update_traces(
                textinfo="label+percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    f"Revenue: {currency_symbol}%{{value:,.2f}}<br>"
                    "Revenue share: %{percent}"
                    "<extra></extra>"
                ),
                sort=False,
            )
            donut.update_layout(
                paper_bgcolor=card_background_rgba,
                plot_bgcolor=card_background_rgba,
                showlegend=True,
                legend_title_text="Product",
                font={"color": card_text},
                margin=dict(l=20, r=20, t=20, b=20),
                annotations=[
                    {
                        "text": f"Total<br>{currency_symbol}{product_revenue:,.0f}",
                        "x": 0.5,
                        "y": 0.5,
                        "showarrow": False,
                        "font": {"size": 16, "color": card_text},
                    }
                ],
            )
            st.plotly_chart(donut, use_container_width=True, config=chart_config)
        else:
            st.info("Add a product column to see product revenue share.")

        st.subheader("Performance highlights")
        highlights = st.columns(2)
        for position, key, title in (
            (0, "best_region", "Top region"),
            (1, "worst_region", "Lowest region"),
            (2, "best_product", "Top product"),
            (3, "worst_product", "Lowest product"),
        ):
            result = metrics[key]
            highlights[position % 2].metric(
                title,
                result["name"] if result else "—",
                f"{currency_symbol}{result['revenue']:,.2f} revenue" if result else None,
            )

    with lower_right:
        st.subheader("AI executive insights")
        if anomalies:
            st.warning(
                "⚠️ **Automated risk alerts**\n\n"
                + "\n".join(f"- {html.escape(item)}" for item in anomalies)
            )
        elif metrics.get("availability", {}).get("revenue", True):
            st.success("No significant revenue drops or low-performing categories detected.")
        else:
            st.info("Automated revenue alerts are unavailable because revenue data was not detected.")

        if st.button("Generate AI insights", type="primary"):
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if api_key:
                try:
                    ai_metrics = {**metrics, "display_currency": currency_symbol}
                    st.session_state["ai_insights"] = generate_insights(
                        ai_metrics,
                        api_key=api_key,
                    )
                    st.session_state["ai_insights"]["source"] = "Gemini (aggregated metrics)"
                    st.session_state.pop("ai_insights_error", None)
                except InsightGenerationError as exc:
                    st.session_state["ai_insights"] = generate_local_insights(
                        {**metrics, "display_currency": currency_symbol}
                    )
                    st.session_state["ai_insights_error"] = (
                        f"Gemini was unavailable ({exc}); showing local analytics instead."
                    )
            else:
                st.session_state["ai_insights"] = generate_local_insights(
                    {**metrics, "display_currency": currency_symbol}
                )
                st.session_state.pop("ai_insights_error", None)

        if st.session_state.get("ai_insights_error"):
            st.warning(st.session_state["ai_insights_error"])

        insights = st.session_state.get("ai_insights")
        if insights:
            st.caption(f"Insight source: {insights.get('source', 'Gemini')}.")
            recommendation_list = "".join(
                f"<li>📌 {html.escape(str(item))}</li>"
                for item in insights["recommendations"]
            )
            risks = insights.get("risks", [])
            risk_list = "".join(
                f"<li>⚠️ {html.escape(str(item))}</li>" for item in risks
            )
            risk_section = (
                f"<strong>Areas of concern</strong><ul>{risk_list}</ul>"
                if risk_list
                else ""
            )
            st.markdown(
                (
                    '<div class="executive-card">'
                    "<h3>Executive summary</h3>"
                    f"<p>{html.escape(str(insights['executive_summary']))}</p>"
                    "<strong>Recommended actions</strong>"
                    f"<ul>{recommendation_list}</ul>"
                    f"{risk_section}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("Generate AI insights to see an executive summary and recommendations.")

        st.markdown("**Ask AI about your data**")
        question_column, voice_column = st.columns([5, 2])
        with voice_column:
            voice_result = voice_assistant(
                mode="input",
                key=f"voice_input_{abs(hash(insight_context))}",
                default={"text": "", "timestamp": 0},
            )
            st.caption(
                "Speech recognition needs a supported browser and microphone permission. "
                "Some browsers use an internet speech service; typing always works."
            )

        question_key = f"ask_question_{abs(hash(insight_context))}"
        if isinstance(voice_result, dict):
            recognized_text = str(voice_result.get("text", "")).strip()
            timestamp = voice_result.get("timestamp")
            if recognized_text and timestamp != st.session_state.get("_last_voice_timestamp"):
                st.session_state[question_key] = recognized_text
                st.session_state["_last_voice_timestamp"] = timestamp

        with question_column:
            with st.form(f"ask_data_form_{abs(hash(insight_context))}"):
                question = st.text_input(
                    "Ask a question about the uploaded dataset",
                    placeholder="Which month had the highest profit?",
                    key=question_key,
                )
                ask_submitted = st.form_submit_button("Ask AI")
        if ask_submitted:
            if not question.strip():
                st.error("Enter a question about the dashboard data.")
            else:
                api_key = os.getenv("GEMINI_API_KEY")
                answer = _answer_data_question(
                    question.strip(),
                    metrics,
                    api_key or "",
                    data=filtered_data,
                )
                st.session_state.setdefault("data_chat_history", []).append(
                    {"question": question.strip(), "answer": answer}
                )

        for message in st.session_state.get("data_chat_history", []):
            st.markdown(f"**You:** {message['question']}")
            st.markdown(f"**AI:** {message['answer']}")
            voice_assistant(
                mode="answer",
                text=message["answer"],
                key=f"voice_answer_{abs(hash((insight_context, message['question'], message['answer'])))}",
            )

    st.subheader("Export report")
    export_format = st.selectbox(
        "Choose a report format",
        ("Markdown", "CSV", "PDF"),
        label_visibility="collapsed",
    )
    if export_format == "Markdown":
        st.download_button(
            "Download Markdown report",
            data=generate_markdown_report(
                metrics,
                insights=insights,
                currency_symbol=currency_symbol,
            ),
            file_name="business_insights_report.md",
            mime="text/markdown",
        )
    elif export_format == "CSV":
        st.download_button(
            "Download CSV report",
            data=_generate_csv_report(metrics, insights),
            file_name="business_insights_report.csv",
            mime="text/csv",
        )
    else:
        st.download_button(
            "Download PDF report",
            data=_generate_pdf_report(metrics, insights, anomalies),
            file_name="business_insights_report.pdf",
            mime="application/pdf",
        )
