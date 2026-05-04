from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.critic import run_critic
from tests._helpers import as_json, build_weak_sample_draft


def test_critic_finds_real_weaknesses():
    weak_draft = build_weak_sample_draft()
    feedback = run_critic(weak_draft, iteration=0)

    # deterministic sanity checks
    assert feedback.verdict == "NEEDS_REVISION"
    assert len(feedback.issues) >= 1

    metric = GEval(
        name="Critic Quality",
        criteria=(
            "Evaluate whether the critic feedback identifies real weaknesses in the draft, "
            "especially unsupported claims, weak sourcing, bias, or missing perspectives. "
            "Good critic feedback should be actionable and specific."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
    )

    test_case = LLMTestCase(
        input=as_json(weak_draft),
        actual_output=as_json(feedback),
    )

    assert_test(test_case, [metric])