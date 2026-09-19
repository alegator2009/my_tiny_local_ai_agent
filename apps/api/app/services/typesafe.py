"""Small, failure-tolerant native client for TypeSafe's Jev model.

Jev is used only for bounded semantic choices.  The application retains a
deterministic fallback for every call, therefore a missing token, timeout, or
unexpected API response can never interrupt a chat turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import TypeSafeConfig

logger = logging.getLogger(__name__)
_SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]


async def evaluate_choice(
    *,
    state: Any,
    instructions: str,
    criteria: dict[str, str | None],
    config: TypeSafeConfig,
) -> ChoiceAnswer | None:
    """Ask Jev one typed Choice question, returning ``None`` on fallback.

    Never log the key, input state, or raw response: chat state may contain
    user data, while the caller only needs a validated selected option.
    """

    if not config.is_ready() or not criteria:
        return None

    payload = {
        "model": config.model or "jev-latest",
        "state": state,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }
    try:
        async with httpx.AsyncClient(timeout=float(config.timeout_sec)) as client:
            response = await client.post(
                _SYSTEM_ONE_URL,
                json=payload,
                headers={"Authorization": f"Bearer {config.api_key.strip()}"},
            )
        if response.status_code >= 400:
            logger.info("TypeSafe Jev unavailable (HTTP %s); using local fallback", response.status_code)
            return None
        body = response.json()
        answer = body.get("answers", {}).get("decision", {}) if isinstance(body, dict) else {}
        choice = answer.get("choice") if isinstance(answer, dict) else None
        if not isinstance(choice, str) or choice not in criteria:
            return None
        raw_probabilities = answer.get("probabilities", {})
        if not isinstance(raw_probabilities, dict):
            raw_probabilities = {}
        probabilities = {
            str(key): float(value)
            for key, value in raw_probabilities.items()
            if isinstance(value, (int, float))
        }
        raw_confidence = answer.get("confidence")
        confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else probabilities.get(choice, 0.0)
        return ChoiceAnswer(choice=choice, confidence=max(0.0, min(confidence, 1.0)), probabilities=probabilities)
    except (httpx.HTTPError, ValueError, TypeError):
        logger.info("TypeSafe Jev request failed; using local fallback")
        return None
