"""
pricing.py
----------
Fetches per-model token pricing from OpenRouter's public /models endpoint
so the app can report an accurate total cost for a batch run, instead of a
hardcoded/stale price table.
"""

from __future__ import annotations

import requests

MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_model_pricing(api_key: str = "") -> dict:
    """Returns {model_id: {"prompt": price_per_token_usd, "completion": price_per_token_usd}}.
    Returns {} on any failure (e.g. offline) rather than raising, since pricing
    is a nice-to-have for the cost report, not a blocker for the pipeline."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(MODELS_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return {}

    out = {}
    for model in data:
        model_id = model.get("id")
        pricing = model.get("pricing", {})
        try:
            out[model_id] = {
                "prompt": float(pricing.get("prompt", 0) or 0),
                "completion": float(pricing.get("completion", 0) or 0),
            }
        except (TypeError, ValueError):
            continue
    return out


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, price: dict | None) -> float | None:
    if not price:
        return None
    p = price.get("prompt", 0) or 0
    c = price.get("completion", 0) or 0
    return (prompt_tokens or 0) * p + (completion_tokens or 0) * c
