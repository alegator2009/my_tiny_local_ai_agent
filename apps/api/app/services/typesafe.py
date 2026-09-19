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
    """Ask Jev one typed Choice question, returning ``None`` on fallback."""

    answers = await evaluate_choices(
        state=state,
        questions={"decision": {"instructions": instructions, "criteria": criteria}},
        config=config,
    )
    return (answers or {}).get("decision")


async def evaluate_choices(
    *,
    state: Any,
    questions: dict[str, dict[str, Any]],
    config: TypeSafeConfig,
) -> dict[str, ChoiceAnswer] | None:
    """Ask Jev independent Choice questions in one System One request.

    Every question is validated against its own closed set of criteria.  A
    missing or malformed answer is omitted so callers can use their existing
    deterministic fallback for that dimension.
    """

    if not config.is_ready() or not questions:
        return None

    typed_questions: dict[str, dict[str, Any]] = {}
    criteria_by_id: dict[str, dict[str, str | None]] = {}
    for question_id, question in questions.items():
        instructions = question.get("instructions")
        criteria = question.get("criteria")
        if not isinstance(question_id, str) or not isinstance(instructions, str) or not isinstance(criteria, dict) or not criteria:
            continue
        normalized_criteria = {
            str(key): value if isinstance(value, str) or value is None else str(value)
            for key, value in criteria.items()
        }
        if not normalized_criteria:
            continue
        criteria_by_id[question_id] = normalized_criteria
        typed_questions[question_id] = {
            "type": "choice",
            "instructions": instructions,
            "criteria": normalized_criteria,
        }

    if not typed_questions:
        return None

    payload = {
        "model": config.model or "jev-latest",
        "state": state,
        "questions": typed_questions,
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
        raw_answers = body.get("answers", {}) if isinstance(body, dict) else {}
        if not isinstance(raw_answers, dict):
            return None
        answers: dict[str, ChoiceAnswer] = {}
        for question_id, criteria in criteria_by_id.items():
            answer = raw_answers.get(question_id, {})
            choice = answer.get("choice") if isinstance(answer, dict) else None
            if not isinstance(choice, str) or choice not in criteria:
                continue
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
            answers[question_id] = ChoiceAnswer(
                choice=choice,
                confidence=max(0.0, min(confidence, 1.0)),
                probabilities=probabilities,
            )
        return answers or None
    except (httpx.HTTPError, ValueError, TypeError):
        logger.info("TypeSafe Jev request failed; using local fallback")
        return None
