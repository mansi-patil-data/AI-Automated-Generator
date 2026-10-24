"""Gemini-powered executive insight generation."""

from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai.errors import APIError


class InsightGenerationError(RuntimeError):
    """Raised when the AI service cannot return valid structured insights."""


def generate_local_insights(metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize measured dashboard metrics without relying on a remote AI service."""
    totals = metrics["totals"]
    availability = metrics.get("availability", {})
    currency = metrics.get("display_currency", "")
    currency_prefix = f"{currency}" if currency else ""
    facts: list[str] = []
    if availability.get("revenue", True):
        facts.append(f"Revenue totals {currency_prefix}{totals['revenue']:,.2f}.")
    if availability.get("expenses", True):
        facts.append(f"Expenses total {currency_prefix}{totals['expenses']:,.2f}.")
    if availability.get("profit", True):
        facts.append(f"Profit totals {currency_prefix}{totals['profit']:,.2f}.")
    if availability.get("profit_margin", True):
        facts.append(f"Profit margin is {totals['profit_margin']:.1f}%.")

    trends = metrics.get("monthly_trends", [])
    if len(trends) > 1 and availability.get("revenue", True):
        previous, current = trends[-2:]
        change = totals.get("mom_growth")
        if change is not None:
            direction = "increased" if change >= 0 else "decreased"
            facts.append(
                f"Revenue {direction} by {abs(change):.1f}% from "
                f"{previous['month']} to {current['month']}."
            )
    for label, key in (
        ("Highest-revenue product", "best_product"),
        ("Lowest-revenue product", "worst_product"),
        ("Highest-revenue region", "best_region"),
        ("Lowest-revenue region", "worst_region"),
    ):
        performer = metrics.get(key)
        if performer:
            facts.append(
                f"{label}: {performer['name']} "
                f"({currency_prefix}{performer['revenue']:,.2f} revenue)."
            )
    if not facts:
        facts.append("The available dataset does not contain recognized business metrics.")

    recommendations = []
    if len(trends) > 1 and availability.get("revenue", True):
        recommendations.append(
            "Review the latest monthly revenue change alongside sales and operational "
            "drivers before adjusting the forecast."
        )
    if metrics.get("worst_product") and metrics.get("best_product"):
        recommendations.append(
            "Compare the lowest- and highest-revenue products to identify testable "
            "assortment or promotion opportunities."
        )
    if metrics.get("worst_region") and metrics.get("best_region"):
        recommendations.append(
            "Review regional performance differences and validate them against "
            "local demand and coverage."
        )
    if not recommendations:
        recommendations.append(
            "Add date, product, or region fields to enable more detailed performance "
            "comparisons."
        )

    risks = [
        item
        for item in facts
        if item.startswith("Lowest-revenue")
        or "decreased by" in item
    ]
    return {
        "executive_summary": " ".join(facts),
        "recommendations": recommendations,
        "risks": risks,
        "source": "Local analytics",
    }


def generate_insights(
    metrics: dict[str, Any],
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    """Send aggregated metrics to Gemini and return structured recommendations."""
    if not api_key.strip():
        raise InsightGenerationError("A Gemini API key is required.")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=(
                "You are a careful business analyst. Use only the supplied "
                "aggregated metrics; do not invent facts. Return one JSON object "
                'with keys "executive_summary" (string), "recommendations" '
                "(array of 3 concise strings), and \"risks\" (array of strings). "
                "Clearly distinguish observations from suggestions.\n\n"
                f"Metrics JSON:\n{json.dumps(metrics, allow_nan=False)}"
            ),
            config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            },
        )
        content = response.text
    except APIError as exc:
        raise InsightGenerationError("Gemini request failed or was rate-limited.") from exc
    except ValueError as exc:
        raise InsightGenerationError(f"Gemini could not process the request: {exc}") from exc
    except Exception as exc:
        raise InsightGenerationError("Gemini request failed.") from exc

    if not content:
        raise InsightGenerationError("Gemini returned an empty response.")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InsightGenerationError("Gemini returned malformed JSON.") from exc

    if (
        not isinstance(result, dict)
        or not isinstance(result.get("executive_summary"), str)
        or not isinstance(result.get("recommendations"), list)
        or not all(isinstance(item, str) for item in result["recommendations"])
        or not isinstance(result.get("risks", []), list)
        or not all(isinstance(item, str) for item in result.get("risks", []))
    ):
        raise InsightGenerationError("Gemini response did not match the expected insight structure.")
    result.setdefault("risks", [])
    return result
