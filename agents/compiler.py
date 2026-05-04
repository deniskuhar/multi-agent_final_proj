from __future__ import annotations

import json

from langchain_openai import ChatOpenAI

from config import get_settings
from langfuse_utils import (
    get_callback_handler,
    observe,
    propagate_attributes,
    safe_metadata,
)
from schemas import DraftReport, FinalReport
from tools import save_report

settings = get_settings()

model = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.1,
    timeout=settings.request_timeout_seconds,
)

COMPILER_PROMPT = """
You are the Report Compiler in a market research multi-agent system.

Your task:
- Take an approved or best-available DraftReport.
- Convert it into a polished FinalReport.
- Do not introduce new claims that are not present in the draft.
- Preserve the core evidence and findings.
- Make recommendations that follow from the draft findings.

Rules:
- Keep the executive summary concise and accurate.
- Key findings should be crisp and decision-useful.
- Recommendations should be actionable.
- Methodology should briefly describe that the report used web search + RAG + critic review.
- Be faithful to the draft and avoid inventing unsupported details.

Return a valid FinalReport object.
""".strip()


def final_report_to_markdown(
    report: FinalReport,
    topic: str,
    scope: str,
    unresolved_critic_issues: list[str] | None = None,
) -> str:
    lines = [
        f"# Market Research Report: {topic}",
        "",
        "## Scope",
        scope,
        "",
        "## Executive Summary",
        report.executive_summary,
        "",
        "## Key Findings",
    ]

    for item in report.key_findings:
        lines.append(f"- {item}")

    lines.extend(["", "## Recommendations"])
    for item in report.recommendations:
        lines.append(f"- {item}")

    lines.extend(["", "## Sources"])
    for item in report.sources:
        lines.append(f"- {item}")

    lines.extend(["", "## Methodology", report.methodology])

    if unresolved_critic_issues:
        lines.extend(["", "## Unresolved Critic Issues"])
        for item in unresolved_critic_issues:
            lines.append(f"- {item}")

    lines.append("")
    return "\n".join(lines)


@observe(name="compiler")
def run_compiler(draft: DraftReport, iteration: int = 0) -> FinalReport:
    prompt = f"""
{COMPILER_PROMPT}

DraftReport:
{json.dumps(draft.model_dump(), ensure_ascii=False, indent=2)}
""".strip()

    with propagate_attributes(
        metadata=safe_metadata(
            {
                "agent": "compiler",
                "iteration": iteration,
                "draftfindings": len(draft.findings),
                "draftsources": len(draft.sources),
            }
        ),
        tags=["market-analyst", "compiler"],
    ):
        structured_model = model.with_structured_output(FinalReport)
        return structured_model.invoke(
            prompt,
            config={"callbacks": [get_callback_handler()]},
        )


def compile_and_save_report(
    *,
    draft: DraftReport,
    topic: str,
    scope: str,
    filename: str = "ev_market_europe_2025_report.md",
    unresolved_critic_issues: list[str] | None = None,
    iteration: int = 0,
) -> tuple[FinalReport, str]:
    final_report = run_compiler(draft, iteration=iteration)

    markdown = final_report_to_markdown(
        final_report,
        topic=topic,
        scope=scope,
        unresolved_critic_issues=unresolved_critic_issues,
    )

    save_result = save_report(filename, markdown)
    return final_report, save_result