"""Tests for Gemini-powered executive insight generation."""

import json

import pytest

from src import insightgenerator
from src.insightgenerator import (
    InsightGenerationError,
    generate_insights,
    generate_local_insights,
)


class FakeGeminiModel:
    response_text = ""
    prompt = ""
    generation_config = {}

    def generate_content(
        self,
        model: str,
        contents: str,
        config: dict,
    ) -> object:
        self.model_name = model
        self.prompt = contents
        self.generation_config = config
        return type("FakeResponse", (), {"text": self.response_text})()

class FakeGeminiClient:
    def __init__(self, api_key: str, fake_model: FakeGeminiModel) -> None:
        self.api_key = api_key
        self.models = fake_model


def test_generate_insights_parses_gemini_json(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = FakeGeminiModel()
    fake_model.response_text = json.dumps(
        {
            "executive_summary": "Revenue increased.",
            "recommendations": ["Review growth drivers."],
            "risks": [],
        }
    )
    monkeypatch.setattr(
        insightgenerator.genai,
        "Client",
        lambda **kwargs: FakeGeminiClient(fake_model=fake_model, **kwargs),
    )

    result = generate_insights({"totals": {"revenue": 100}}, api_key="test-key")

    assert result == {
        "executive_summary": "Revenue increased.",
        "recommendations": ["Review growth drivers."],
        "risks": [],
    }
    assert "Metrics JSON" in fake_model.prompt
    assert fake_model.generation_config["response_mime_type"] == "application/json"


def test_generate_insights_requires_api_key() -> None:
    with pytest.raises(InsightGenerationError, match="Gemini API key"):
        generate_insights({"totals": {}}, api_key=" ")


def test_generate_insights_rejects_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = FakeGeminiModel()
    fake_model.response_text = '{"executive_summary": 4}'
    monkeypatch.setattr(
        insightgenerator.genai,
        "Client",
        lambda **kwargs: FakeGeminiClient(fake_model=fake_model, **kwargs),
    )

    with pytest.raises(InsightGenerationError, match="expected insight structure"):
        generate_insights({"totals": {}}, api_key="test-key")


def test_local_insights_only_report_available_metrics() -> None:
    result = generate_local_insights(
        {
            "totals": {
                "revenue": 125.0,
                "expenses": 0.0,
                "profit": 0.0,
                "profit_margin": 0.0,
                "mom_growth": None,
            },
            "availability": {
                "revenue": True,
                "expenses": False,
                "profit": False,
                "profit_margin": False,
            },
            "monthly_trends": [],
            "best_product": None,
            "worst_product": None,
            "best_region": None,
            "worst_region": None,
        }
    )
    assert "Revenue totals 125.00." in result["executive_summary"]
    assert "Profit totals" not in result["executive_summary"]
    assert result["source"] == "Local analytics"
