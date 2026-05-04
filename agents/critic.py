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
from schemas import CriticFeedback, DraftReport
from tools import trusted_web_search

settings = get_settings()

model = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.1,
    timeout=settings.request_timeout_seconds,
)

CRITIC_PROMPT = """
You are the Critic in a market research multi-agent system.

Your role:
- Review the draft report critically.
- Act as a devil's advocate.
- Identify weak evidence, unsupported claims, bias, and missing perspectives.
- Decide whether the draft is good enough or needs revision.

Rules:
- Be specific and concrete.
- Do not rewrite the report.
- Focus on evidence quality, balance, missing risks, and factual support.
- If the report lacks source precision, say so clearly.
- Prefer NEEDS_REVISION unless the report is clearly evidence-backed and balanced.
- Use the full score range from 0.0 to 1.0.
- Reserve 0.0 only for unusable drafts with severe failures.
- A draft with useful structure but weak source precision should usually score between 0.3 and 0.6.
- If the revised draft improves source precision or coverage, reflect that in the score.

Return a valid CriticFeedback object.
""".strip()


def build_critic_input(draft: DraftReport) -> str:
    findings_for_check = []
    for item in draft.findings[:5]:
        findings_for_check.append(f"- {item.title}: {item.insight}")

    verification_queries = [
        "EV market in Europe 2025 official BEV market share data",
        "EV charging infrastructure Europe 2025 official data",
        "EU EV regulation 2035 CO2 target official source",
    ]

    verification_results = []
    for q in verification_queries:
        verification_results.append(
            f"Query: {q}\n{trusted_web_search(q, max_results=8, keep_results=4)}"
        )

    return f"""
Draft report JSON:
{json.dumps(draft.model_dump(), ensure_ascii=False, indent=2)}

Key findings to stress-test:
{chr(10).join(findings_for_check)}

Independent verification context:
{chr(10).join(verification_results)}
""".strip()


@observe(name="critic")
def run_critic(draft: DraftReport, iteration: int = 0) -> CriticFeedback:
    prompt = f"""
{CRITIC_PROMPT}

{build_critic_input(draft)}
""".strip()

    with propagate_attributes(
        metadata=safe_metadata(
            {
                "agent": "critic",
                "iteration": iteration,
                "draftfindings": len(draft.findings),
                "draftsources": len(draft.sources),
            }
        ),
        tags=["market-analyst", "critic"],
    ):
        structured_model = model.with_structured_output(CriticFeedback)
        return structured_model.invoke(
            prompt,
            config={"callbacks": [get_callback_handler()]},
        )