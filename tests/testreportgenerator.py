import pandas as pd

from src.dataanalyzer import analyze_data
from src.reportgenerator import generate_markdown_report


def test_markdown_report_labels_unavailable_metrics_and_uses_selected_currency() -> None:
    metrics = analyze_data(pd.DataFrame({"revenue": [125.0]}))

    report = generate_markdown_report(metrics, currency_symbol="$")

    assert "**Revenue:** $125.00" in report
    assert "**Expenses:** Not available" in report
    assert "**Profit:** Not available" in report
    assert "**Profit margin:** Not available" in report
