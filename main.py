from __future__ import annotations

import json

from config import get_settings
from graph import run_pipeline

settings = get_settings()


def main() -> None:
    print("=" * 68)
    print("Market Analyst Multi-Agent System")
    print("=" * 68)

    topic = input(f"Topic [{settings.topic}]: ").strip() or settings.topic
    scope = input(f"Scope [{settings.scope}]: ").strip() or settings.scope

    final_state = run_pipeline(
        topic=topic,
        scope=scope,
        focus_areas=settings.focus_areas,
        user_id=getattr(settings, "langfuse_default_user_id", "denys"),
    )

    print("\n" + "=" * 68)
    print("FINAL STATE")
    print("=" * 68)

    if final_state.get("critic_feedback"):
        print("\nCritic feedback:")
        print(json.dumps(final_state["critic_feedback"], ensure_ascii=False, indent=2))

    if final_state.get("final_report"):
        print("\nFinal report:")
        print(json.dumps(final_state["final_report"], ensure_ascii=False, indent=2))

    if final_state.get("saved_report_path"):
        print("\nSaved:")
        print(final_state["saved_report_path"])

    if final_state.get("session_id"):
        print("\nSession ID:")
        print(final_state["session_id"])


if __name__ == "__main__":
    main()