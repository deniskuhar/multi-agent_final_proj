from __future__ import annotations

import re
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler

load_dotenv()


def get_langfuse_client():
    return get_client()


def get_callback_handler() -> CallbackHandler:
    # In Langfuse v4, use a plain CallbackHandler() and propagate trace attrs
    # via propagate_attributes() instead of old update_trace patterns.
    return CallbackHandler()


def safe_metadata(values: dict[str, object] | None = None) -> dict[str, str]:
    """Normalize metadata for Langfuse:
    - keys: alphanumeric only
    - values: strings up to 200 chars
    """
    output: dict[str, str] = {}
    if not values:
        return output

    for key, value in values.items():
        clean_key = re.sub(r"[^A-Za-z0-9]", "", str(key))
        if not clean_key or value is None:
            continue

        clean_value = str(value)[:200]
        output[clean_key] = clean_value

    return output


__all__ = [
    "observe",
    "propagate_attributes",
    "get_langfuse_client",
    "get_callback_handler",
    "safe_metadata",
]