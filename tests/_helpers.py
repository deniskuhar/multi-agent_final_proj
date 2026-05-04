from __future__ import annotations

import json
from pathlib import Path

from config import get_settings
from graph import run_pipeline
from schemas import DraftReport, Finding


def as_json(data) -> str:
    if hasattr(data, "model_dump"):
        return json.dumps(data.model_dump(), ensure_ascii=False, indent=2)
    return json.dumps(data, ensure_ascii=False, indent=2)


def default_topic() -> str:
    return "EV market in Europe 2025"


def default_scope() -> str:
    return "Passenger EV adoption, charging infrastructure, regulation, competition, risks and outlook"


def default_focus_areas() -> list[str]:
    settings = get_settings()
    return settings.focus_areas


def build_strong_sample_draft() -> DraftReport:
    return DraftReport(
        executive_summary=(
            "The European EV market continues to expand, but growth is uneven across countries. "
            "Charging infrastructure, regulation, competition, and consumer affordability remain key drivers."
        ),
        findings=[
            Finding(
                title="BEV market share growth",
                insight=(
                    "BEV adoption in Europe is increasing, although market-share estimates vary by source and geography."
                ),
                evidence=[
                    "ICCT European Market Monitor, December 2025: BEV share continued rising, but country-level trends varied.",
                    "IEA Global EV Outlook 2025: Europe remains one of the largest EV markets globally."
                ],
                risk_level="medium",
            ),
            Finding(
                title="Charging infrastructure disparities",
                insight=(
                    "Public charging infrastructure has expanded, but availability remains uneven across Europe."
                ),
                evidence=[
                    "IEA Global EV Outlook 2025 — charging section: infrastructure deployment remains uneven across countries.",
                    "EAFO Data Update, November 2025: charging-point availability differs significantly across member states."
                ],
                risk_level="high",
            ),
            Finding(
                title="Regulation as a growth driver",
                insight=(
                    "EU emissions regulation continues to support EV adoption, though implementation effects vary across markets."
                ),
                evidence=[
                    "European Commission AFIR / clean transport policy pages: regulatory framework supports transition.",
                    "ACEA CO2 targets roadmap: automakers face major compliance pressure."
                ],
                risk_level="medium",
            ),
            Finding(
                title="Intensifying competition",
                insight=(
                    "Competitive pressure is increasing as both legacy automakers and newer entrants compete on price and scale."
                ),
                evidence=[
                    "S&P Global EV whitepaper: competition in the EV market is intensifying.",
                    "ICCT market monitoring and European market reports indicate shifting competitive dynamics."
                ],
                risk_level="medium",
            ),
            Finding(
                title="Consumer barriers remain",
                insight=(
                    "High upfront cost, charging convenience, and economic uncertainty still constrain broader adoption."
                ),
                evidence=[
                    "Deloitte 2025 Global Automotive Consumer Study: affordability and charging remain key barriers.",
                    "EAFO Consumer Monitor 2025: consumer sentiment remains sensitive to infrastructure access and cost."
                ],
                risk_level="high",
            ),
        ],
        sources=[
            "ICCT European Market Monitor, December 2025",
            "IEA Global EV Outlook 2025",
            "EAFO Data Update, November 2025",
            "European Commission AFIR policy pages",
            "ACEA CO2 targets roadmap",
            "Deloitte 2025 Global Automotive Consumer Study",
        ],
        data_points=[
            "BEV adoption continued rising in Europe in 2025, though estimates varied by source.",
            "Charging infrastructure expansion remained uneven across Europe.",
            "Affordability and charging availability remained major adoption barriers."
        ],
    )


def build_weak_sample_draft() -> DraftReport:
    return DraftReport(
        executive_summary="The EV market is growing quickly and everything looks positive.",
        findings=[
            Finding(
                title="Market growth",
                insight="Europe will definitely dominate EV adoption and the market will rise sharply.",
                evidence=["Some industry reports suggest strong growth."],
                risk_level="low",
            ),
            Finding(
                title="Consumer demand",
                insight="Consumers clearly prefer EVs now and barriers are disappearing.",
                evidence=["General market sentiment is positive."],
                risk_level="low",
            ),
        ],
        sources=[
            "Industry blog",
            "General market commentary",
        ],
        data_points=[
            "The market is growing fast."
        ],
    )


def run_e2e_once():
    settings = get_settings()
    state = run_pipeline(
        topic=default_topic(),
        scope=default_scope(),
        focus_areas=settings.focus_areas,
        user_id=getattr(settings, "langfuse_default_user_id", "denys"),
    )
    return state


def saved_report_exists(saved_report_path: str | None) -> bool:
    if not saved_report_path:
        return False
    # save_report returns "Report saved to <path>"
    prefix = "Report saved to "
    if saved_report_path.startswith(prefix):
        path = saved_report_path[len(prefix):]
    else:
        path = saved_report_path
    return Path(path).exists()