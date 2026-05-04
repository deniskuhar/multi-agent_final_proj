from __future__ import annotations

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.compiler import run_compiler
from tests._helpers import as_json, build_strong_sample_draft


def test_compiler_preserves_structure_and_meaning():
    strong_draft = build_strong_sample_draft()
    final_report = run_compiler(strong_draft, iteration=0)

    assert final_report.executive_summary.strip()
    assert len(final_report.key_findings) >= 3
    assert len(final_report.recommendations) >= 3
    assert len(final_report.sources) >= 3
    assert final_report.methodology.strip()

    metric = GEval(
        name="Compiler Quality",
        criteria=(
            "Evaluate whether the final report is well-structured, preserves the core meaning of the draft, "
            "and includes the required sections: executive summary, key findings, recommendations, sources, and methodology."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
    )

    test_case = LLMTestCase(
        input=as_json(strong_draft),
        actual_output=as_json(final_report),
    )

    assert_test(test_case, [metric])