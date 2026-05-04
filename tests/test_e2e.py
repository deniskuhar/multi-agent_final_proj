from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

import graph
from tests._helpers import (
    as_json,
    default_scope,
    default_topic,
    run_e2e_once,
    saved_report_exists,
)


def test_e2e_market_report_relevance_and_balance():
    old_max = graph.settings.max_iterations
    graph.settings.max_iterations = 4
    try:
        final_state = run_e2e_once()
    finally:
        graph.settings.max_iterations = old_max

    assert final_state.get("final_report") is not None
    assert saved_report_exists(final_state.get("saved_report_path"))

    final_report = final_state["final_report"]

    metric = GEval(
        name="E2E Report Quality",
        criteria=(
            "Evaluate whether the final market research output answers the requested topic and scope, "
            "is reasonably balanced, includes relevant findings and recommendations, and reads like a useful final report."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
    )

    test_case = LLMTestCase(
        input=f"Topic: {default_topic()}\nScope: {default_scope()}",
        actual_output=as_json(final_report),
    )

    assert_test(test_case, [metric])