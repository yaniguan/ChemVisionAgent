"""Shared JSON-extraction and type-coercion utilities for skill parsers."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from raw model output.

    Tries, in order:
    1. Direct ``json.loads`` on the full text.
    2. Content inside a fenced code block (````json … ````).
    3. The first ``{…}`` span found by regex.

    Returns
    -------
    dict | None
        Parsed dict on success, ``None`` on complete failure.
    """
    # 1. Direct parse
    stripped = text.strip()
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Fenced code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        try:
            result = json.loads(m.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 3. First balanced brace span
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def to_float(value: Any, default: float | None = None) -> float | None:
    """Coerce *value* to float, returning *default* on failure or None input."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_float_required(value: Any, default: float = 0.0) -> float:
    """Coerce *value* to float, returning *default* when coercion fails."""
    result = to_float(value, default)
    return result if result is not None else default


def to_str(value: Any, default: str = "") -> str:
    """Coerce *value* to str safely."""
    if value is None:
        return default
    return str(value)


def to_list(value: Any) -> list:
    """Return *value* as a list, or an empty list if it is not a list."""
    return value if isinstance(value, list) else []
