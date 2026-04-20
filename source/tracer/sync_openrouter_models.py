#!/usr/bin/env python3
"""Download OpenRouter model pricing and save to model_prices.json.

Usage:
    python3 sync_openrouter_models.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent
MODEL_PRICES_FILE = REPO_ROOT / "model_prices.json"


def fetch_openrouter_models() -> list[dict]:
    resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=15)
    resp.raise_for_status()
    return resp.json()["data"]


def main() -> None:
    print("Fetching models from OpenRouter...")
    models = fetch_openrouter_models()
    print(f"Found {len(models)} models")

    prices: dict[str, dict] = {}
    for model in models:
        model_id = model["id"]
        pricing = model.get("pricing", {})
        input_price  = float(pricing.get("prompt",     "0") or "0")
        output_price = float(pricing.get("completion", "0") or "0")
        prices[f"openrouter/{model_id}"] = {
            "name":           model.get("name", model_id),
            "context_length": model.get("context_length"),
            "input_per_token":  input_price,
            "output_per_token": output_price,
            "input_per_1m":     round(input_price  * 1_000_000, 6),
            "output_per_1m":    round(output_price * 1_000_000, 6),
            "free":             input_price == 0 and output_price == 0,
        }

    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "source":     "https://openrouter.ai/api/v1/models",
        "count":      len(prices),
        "models":     prices,
    }
    MODEL_PRICES_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved → {MODEL_PRICES_FILE}")


if __name__ == "__main__":
    main()
