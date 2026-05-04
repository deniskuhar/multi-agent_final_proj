from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from ddgs import DDGS

from config import get_settings
from retriever import get_retriever

settings = get_settings()

TRUSTED_DOMAINS = [
    "iea.org",
    "acea.auto",
    "theicct.org",
    "alternative-fuels-observatory.ec.europa.eu",
    "transport.ec.europa.eu",
    "deloitte.com",
    "eea.europa.eu",
    "emobilityeurope.org",
    "ec.europa.eu",
]


def _truncate(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _is_trusted_domain(url: str, trusted_domains: Iterable[str]) -> bool:
    domain = _domain_of(url)
    for trusted in trusted_domains:
        trusted = trusted.lower().replace("www.", "")
        if domain == trusted or domain.endswith("." + trusted):
            return True
    return False


def web_search(query: str, max_results: int = 5) -> str:
    """Search the open web for market information and return formatted results."""
    results: list[str] = []

    try:
        with DDGS(timeout=15) as ddgs:
            for idx, item in enumerate(ddgs.text(query, max_results=max_results), start=1):
                title = item.get("title", "").strip()
                url = item.get("href", "").strip()
                body = item.get("body", "").strip()
                snippet = _truncate(body, 300)
                results.append(f"{idx}. {title}\nURL: {url}\nSnippet: {snippet}")
    except Exception as exc:
        return f"Web search failed for query '{query}': {exc}"

    if not results:
        return f"No web results found for query: {query}"

    return "\n\n".join(results)


def trusted_web_search(
    query: str,
    max_results: int = 8,
    keep_results: int = 5,
    trusted_domains: list[str] | None = None,
) -> str:
    """Search the web and keep only results from trusted domains."""
    trusted_domains = trusted_domains or TRUSTED_DOMAINS
    kept: list[str] = []

    try:
        with DDGS(timeout=15) as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return f"Trusted web search failed for query '{query}': {exc}"

    for item in raw_results:
        title = item.get("title", "").strip()
        url = item.get("href", "").strip()
        body = item.get("body", "").strip()

        if not url or not _is_trusted_domain(url, trusted_domains):
            continue

        snippet = _truncate(body, 320)
        kept.append(
            f"{len(kept)+1}. {title}\n"
            f"URL: {url}\n"
            f"Domain: {_domain_of(url)}\n"
            f"Snippet: {snippet}"
        )
        if len(kept) >= keep_results:
            break

    if not kept:
        return (
            f"No trusted web results found for query: {query}\n"
            f"Trusted domains: {', '.join(trusted_domains)}"
        )

    return "\n\n".join(kept)


def knowledge_search(query: str, k: int = 5) -> str:
    """Search the local RAG knowledge base and return formatted chunks."""
    try:
        retriever = get_retriever()
        return retriever.formatted_search(query, k=k)
    except Exception as exc:
        return f"Knowledge search failed for query '{query}': {exc}"


def save_report(filename: str, content: str) -> str:
    """Save the final markdown report to the output directory."""
    output_dir: Path = settings.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = filename.strip().replace(" ", "_")
    if not safe_name.endswith(".md"):
        safe_name += ".md"

    path = output_dir / safe_name
    path.write_text(content, encoding="utf-8")

    return f"Report saved to {path}"