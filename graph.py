from __future__ import annotations

import json
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agents.analyst import run_analyst
from agents.compiler import compile_and_save_report
from agents.critic import run_critic
from config import get_settings
from langfuse_utils import observe, propagate_attributes, safe_metadata
from schemas import CriticFeedback, DraftReport

settings = get_settings()


class MarketResearchState(TypedDict, total=False):
    topic: str
    scope: str
    focus_areas: list[str]

    draft_report: dict[str, Any] | None
    critic_feedback: dict[str, Any] | None
    final_report: dict[str, Any] | None

    saved_report_path: str | None
    iteration: int
    changes_made: list[str]
    unresolved_critic_issues: list[str]

    session_id: str | None
    user_id: str | None


def analyst_node(state: MarketResearchState) -> MarketResearchState:
    print("=" * 60)
    print(f"[Analyst] iteration={state.get('iteration', 0)}")
    print(f"[Analyst] topic={state.get('topic')}")
    print("=" * 60)

    critic_feedback = state.get("critic_feedback")
    previous_draft = state.get("draft_report")

    critic_issues: list[str] | None = None
    missing_perspectives: list[str] | None = None

    if critic_feedback:
        critic_issues = critic_feedback.get("issues") or []
        missing_perspectives = critic_feedback.get("missing_perspectives") or []

    previous_draft_text: str | None = None
    if previous_draft:
        previous_draft_text = json.dumps(previous_draft, ensure_ascii=False, indent=2)

    draft: DraftReport = run_analyst(
        topic=state["topic"],
        scope=state["scope"],
        focus_areas=state["focus_areas"],
        critic_issues=critic_issues,
        missing_perspectives=missing_perspectives,
        previous_draft=previous_draft_text,
        iteration=state.get("iteration", 0),
    )

    changes_made = list(state.get("changes_made", []))
    if critic_feedback:
        changes_made.append(
            f"Revision round {state.get('iteration', 0)} addressed issues: "
            + "; ".join(critic_issues or [])
        )

    print(f"[Analyst] produced draft with {len(draft.findings)} findings and {len(draft.sources)} sources")

    return {
        "draft_report": draft.model_dump(),
        "changes_made": changes_made,
    }


def critic_node(state: MarketResearchState) -> MarketResearchState:
    print("=" * 60)
    print(f"[Critic] reviewing draft at iteration={state.get('iteration', 0)}")
    print("=" * 60)

    draft_dict = state.get("draft_report")
    if not draft_dict:
        raise ValueError("critic_node requires draft_report in state")

    draft = DraftReport.model_validate(draft_dict)
    feedback: CriticFeedback = run_critic(
        draft,
        iteration=state.get("iteration", 0),
    )

    print(f"[Critic] verdict={feedback.verdict}, score={feedback.score}")
    if feedback.issues:
        print("[Critic] issues:")
        for issue in feedback.issues:
            print(f"  - {issue}")

    return {
        "critic_feedback": feedback.model_dump(),
        "iteration": state.get("iteration", 0) + 1,
    }


def compiler_node(state: MarketResearchState) -> MarketResearchState:
    print("=" * 60)
    print("[Compiler] compiling final report")
    print("=" * 60)

    draft_dict = state.get("draft_report")
    if not draft_dict:
        raise ValueError("compiler_node requires draft_report in state")

    draft = DraftReport.model_validate(draft_dict)

    feedback_dict = state.get("critic_feedback") or {}
    unresolved_issues: list[str] = []

    if feedback_dict.get("verdict") != "APPROVED":
        unresolved_issues = feedback_dict.get("issues") or []

    filename = (
        state["topic"]
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )[:80] + ".md"

    final_report, save_result = compile_and_save_report(
        draft=draft,
        topic=state["topic"],
        scope=state["scope"],
        filename=filename,
        unresolved_critic_issues=unresolved_issues,
        iteration=state.get("iteration", 0),
    )

    print(f"[Compiler] {save_result}")
    if unresolved_issues:
        print("[Compiler] unresolved critic issues included in final markdown")

    return {
        "final_report": final_report.model_dump(),
        "saved_report_path": save_result,
        "unresolved_critic_issues": unresolved_issues,
    }


def route_after_critic(state: MarketResearchState) -> Literal["analyst", "compiler"]:
    feedback_dict = state.get("critic_feedback")
    if not feedback_dict:
        print("[Router] no critic feedback found -> compiler")
        return "compiler"

    verdict = feedback_dict.get("verdict")
    iteration = state.get("iteration", 0)

    print(f"[Router] verdict={verdict}, iteration={iteration}, max_iterations={settings.max_iterations}")

    if verdict == "NEEDS_REVISION" and iteration < settings.max_iterations:
        print("[Router] sending back to analyst for revision")
        return "analyst"

    print("[Router] sending to compiler")
    return "compiler"


def build_graph():
    graph = StateGraph(MarketResearchState)

    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("compiler", compiler_node)

    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "analyst": "analyst",
            "compiler": "compiler",
        },
    )
    graph.add_edge("compiler", END)

    return graph.compile()


research_graph = build_graph()


@observe(name="market-research-run")
def run_pipeline(
    *,
    topic: str,
    scope: str,
    focus_areas: list[str],
    user_id: str = "denys",
    session_id: str | None = None,
) -> MarketResearchState:
    session_id = session_id or str(uuid4())

    with propagate_attributes(
        session_id=session_id,
        user_id=user_id,
        tags=["market-analyst", "evaluator-optimizer", "ev-market"],
        metadata=safe_metadata(
            {
                "project": "market-analyst",
                "topic": topic,
                "scope": scope[:150],
            }
        ),
    ):
        final_state: MarketResearchState = research_graph.invoke(
            {
                "topic": topic,
                "scope": scope,
                "focus_areas": focus_areas,
                "iteration": 0,
                "changes_made": [],
                "draft_report": None,
                "critic_feedback": None,
                "final_report": None,
                "saved_report_path": None,
                "unresolved_critic_issues": [],
                "session_id": session_id,
                "user_id": user_id,
            }
        )

    return final_state