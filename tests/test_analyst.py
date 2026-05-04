from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.analyst import run_analyst
from config import get_settings
from tests._helpers import as_json, default_scope, default_topic


def test_analyst_grounded_and_specific():
    settings = get_settings()

    draft = run_analyst(
        topic=default_topic(),
        scope=default_scope(),
        focus_areas=settings.focus_areas,
        iteration=0,
    )

    # cheap structural checks before judge
    assert len(draft.findings) >= 4
    assert len(draft.sources) >= 4
    assert draft.executive_summary.strip()

    metric = GEval(
        name="Analyst Quality",
        criteria=(
            "Evaluate whether the market research draft is concrete, evidence-backed, and non-generic. "
            "The output should contain specific findings, explicit evidence, and meaningful sources."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
    )

    test_case = LLMTestCase(
        input=f"Topic: {default_topic()}\nScope: {default_scope()}",
        actual_output=as_json(draft),
    )

    assert_test(test_case, [metric])