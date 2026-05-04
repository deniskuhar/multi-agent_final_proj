from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import get_settings
from langfuse_utils import (
    get_callback_handler,
    observe,
    propagate_attributes,
    safe_metadata,
)
from schemas import DraftReport
from tools import TRUSTED_DOMAINS, knowledge_search, trusted_web_search

settings = get_settings()

model = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.1,
    timeout=settings.request_timeout_seconds,
)

ANALYST_PROMPT = """
You are a Research Analyst in a market research multi-agent system.

Your task:
- Research the assigned market topic.
- Use both local RAG evidence and trusted web evidence.
- Produce a structured DraftReport.
- Be concrete, evidence-backed, and avoid generic statements.
- Use critique feedback to improve the draft when provided.

Critical rules:
- Do not invent facts, numbers, or projections.
- If a number is not clearly supported by the provided context, omit it.
- If different sources provide different estimates, explicitly say that estimates vary by source.
- Do not present uncertain forecasts as settled facts.
- Prefer cautious wording such as:
  "is expected to continue increasing",
  "estimates vary by source",
  "multiple sources suggest growth",
  instead of a single hard forecast.
- Prefer local RAG evidence and trusted official sources first.
- Every major finding must be backed by specific evidence.
- Use specific source names: report titles, organizations, or URLs.
- Avoid vague references like "industry reports" or "some sources say".
- Include opportunities, risks, and uncertainties.
- Include at least one counterargument, challenge, or downside risk.
- If critic feedback mentions missing perspectives, explicitly address them.

Evidence style requirements:
- In findings[*].evidence, cite evidence as short statements with source attribution.
- Good example:
  "IEA Global EV Outlook 2025 — charging section: charging expansion remains uneven across Europe."
- Good example:
  "ICCT European Market Monitor, Dec 2025: BEV share continues to rise, but country-level trends vary."
- Bad example:
  "Experts say the market will grow rapidly."

Output expectations:
- executive_summary should be balanced and not overstate certainty.
- findings should be specific and evidence-backed.
- sources should list only specific source names, report titles, or URLs actually used.
- data_points should contain only directly supported metrics.

Return a valid DraftReport object.
""".strip()


def _dedupe(items: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
        if limit is not None and len(result) >= limit:
            break
    return result


def build_analyst_input(
    *,
    topic: str,
    scope: str,
    focus_areas: list[str],
    critic_issues: list[str] | None = None,
    missing_perspectives: list[str] | None = None,
    previous_draft: str | None = None,
) -> str:
    lines = [
        f"Topic: {topic}",
        f"Scope: {scope}",
        "",
        "Focus areas:",
    ]

    for item in focus_areas:
        lines.append(f"- {item}")

    if critic_issues:
        lines.extend(["", "Critic issues to address in this revision:"])
        for item in critic_issues:
            lines.append(f"- {item}")

    if missing_perspectives:
        lines.extend(["", "Missing perspectives that must be included:"])
        for item in missing_perspectives:
            lines.append(f"- {item}")

    if previous_draft:
        lines.extend(
            [
                "",
                "Previous draft (for revision context only):",
                previous_draft[:7000],
            ]
        )

    lines.extend(
        [
            "",
            "Revision instructions:",
            "- Improve source precision.",
            "- Tie claims to specific evidence.",
            "- Add risks, limitations, and counterarguments.",
            "- Remove unsupported numeric claims.",
            "- Prefer trusted and official sources.",
        ]
    )

    return "\n".join(lines)


def build_queries(
    *,
    topic: str,
    scope: str,
    focus_areas: list[str],
    critic_issues: list[str] | None = None,
    missing_perspectives: list[str] | None = None,
) -> list[str]:
    queries = [
        f"{topic} Europe 2025 market size market share sales official data",
        f"{topic} Europe 2025 charging infrastructure public charge points official data",
        f"{topic} Europe 2025 regulation CO2 targets AFIR official source",
        f"{topic} Europe 2025 competition Tesla BYD Volkswagen official market data",
        f"{topic} Europe 2025 risks consumer adoption barriers official report",
    ]

    for area in focus_areas[:5]:
        queries.append(f"{topic} {area} Europe 2025 official report")

    if critic_issues:
        for issue in critic_issues[:4]:
            queries.append(f"{topic} {issue} Europe 2025 official source")

    if missing_perspectives:
        for perspective in missing_perspectives[:4]:
            queries.append(f"{topic} {perspective} Europe 2025 official report")

    return _dedupe(queries, limit=8)


def collect_context(queries: list[str]) -> tuple[str, str]:
    web_context_parts: list[str] = []
    rag_context_parts: list[str] = []

    for q in queries:
        web_context_parts.append(
            f"Query: {q}\n"
            f"{trusted_web_search(q, max_results=10, keep_results=5, trusted_domains=TRUSTED_DOMAINS)}"
        )
        rag_context_parts.append(
            f"Query: {q}\n"
            f"{knowledge_search(q, k=5)}"
        )

    return "\n\n".join(web_context_parts), "\n\n".join(rag_context_parts)


@observe(name="analyst")
def run_analyst(
    *,
    topic: str,
    scope: str,
    focus_areas: list[str],
    critic_issues: list[str] | None = None,
    missing_perspectives: list[str] | None = None,
    previous_draft: str | None = None,
    iteration: int = 0,
) -> DraftReport:
    queries = build_queries(
        topic=topic,
        scope=scope,
        focus_areas=focus_areas,
        critic_issues=critic_issues,
        missing_perspectives=missing_perspectives,
    )

    web_context, rag_context = collect_context(queries)

    analyst_input = build_analyst_input(
        topic=topic,
        scope=scope,
        focus_areas=focus_areas,
        critic_issues=critic_issues,
        missing_perspectives=missing_perspectives,
        previous_draft=previous_draft,
    )

    prompt = f"""
{ANALYST_PROMPT}

User request and scope:
{analyst_input}

Trusted web domains:
{", ".join(TRUSTED_DOMAINS)}

Search queries used:
{chr(10).join(f"- {q}" for q in queries)}

Trusted web search context:
{web_context}

Local RAG context:
{rag_context}

Important drafting constraints:
- Use only information supported by the provided contexts.
- Prefer the local RAG corpus and trusted domains above all other signals.
- If you mention a number, include supporting evidence in the relevant finding.
- If a number is disputed or varies by source, state that estimates vary.
- In `sources`, include only specific source names or URLs actually present in the context.
- Prefer full source labels, for example:
  "ICCT European Market Monitor, December 2025"
  "IEA Global EV Outlook 2025"
  "Transport & Environment EV Progress Report, September 2025"
- Do not use vague source labels such as:
  "New Automotive report"
  "industry report"
  "market sources"
- In `findings[*].evidence`, include short evidence statements with explicit source attribution.
- Include both upside and downside perspectives.
- Include at least one finding or paragraph that reflects uncertainty, constraint, or downside risk.
- Avoid unsupported forecasting language.

Return a valid DraftReport object.
""".strip()

    with propagate_attributes(
        metadata=safe_metadata(
            {
                "agent": "analyst",
                "iteration": iteration,
                "topic": topic,
                "scope": scope[:120],
            }
        ),
        tags=["market-analyst", "analyst"],
    ):
        structured_model = model.with_structured_output(DraftReport)
        return structured_model.invoke(
            prompt,
            config={"callbacks": [get_callback_handler()]},
        )