from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
import json
import re
import uuid
from typing import Any

from ..db import execute, fetch_all, fetch_one, utcnow_iso
from ..storage import (
    append_transcript_event,
    read_memory_file,
    write_checkpoint_file,
    write_memory_file,
)
from .skill_state import (
    apply_step as _skill_state_apply_step,
    build_prompt_bundle as _skill_state_build_prompt_bundle,
    list_states as _skill_state_list_states,
    load_state as _skill_state_load_state,
    plan_skill_delegation as _skill_state_plan_skill_delegation,
    record_skill_tool_observation as _skill_state_record_skill_tool_observation,
    reset_state as _skill_state_reset_state,
    start_or_resume as _skill_state_start_or_resume,
    TransitionError as _SkillTransitionError,
)


SCHEDULED_LINT_INTERVAL_MESSAGES = 8
SCHEDULED_LINT_MIN_SECONDS = 15 * 60
FACT_DECAY_PER_DAY = 0.01
FACT_CONFIDENCE_FLOOR = 0.15


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _recent_messages(session_id: str, limit: int = 20):
    rows = fetch_all(
        "SELECT * FROM messages WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
        (session_id, limit),
    )
    rows.reverse()
    return rows


def update_working_set(session_id: str) -> dict[str, Any]:
    recent = _recent_messages(session_id, limit=20)
    last_user = next((r["content_text"] for r in reversed(recent) if r["role"] == "user"), "")
    last_assistant = next((r["content_text"] for r in reversed(recent) if r["role"] == "assistant"), "")

    # When the most recent user message came from a background run
    # (source='run') we want the *task* as the current objective,
    # not the raw prompt. The task is what the worker is actually
    # executing and the model needs that framing to stay on track.
    last_user_source = next(
        (r["source"] for r in reversed(recent) if r["role"] == "user"),
        None,
    )
    if last_user_source == "run":
        run_row = fetch_one(
            "SELECT task_text FROM runs WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        )
        if run_row and run_row["task_text"]:
            last_user = run_row["task_text"]

    workspace_rows = fetch_all(
        "SELECT * FROM workspace_events WHERE session_id=? ORDER BY timestamp DESC LIMIT 5",
        (session_id,),
    )
    active_files = []
    recent_blockers = []
    for row in workspace_rows:
        payload = json.loads(row["payload_json"])
        if "file" in payload:
            active_files.append(payload["file"])
        if row["event_type"] in {"build_error", "test_result"}:
            recent_blockers.append(row["summary_text"])

    # Surface the active run's progress so the model knows the
    # current step number, the workflow type, and whether a step
    # is currently retrying.
    progress_row = fetch_one(
        """
        SELECT progress_json
        FROM runs
        WHERE session_id=?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (session_id,),
    )
    current_subtask = "Respond to latest user intent with context continuity"
    next_suggested_step = "Continue implementation based on latest user message"
    if progress_row and progress_row["progress_json"]:
        try:
            progress = json.loads(progress_row["progress_json"] or "{}")
        except Exception:
            progress = {}
        phase = progress.get("phase") or "idle"
        cur = progress.get("current_step")
        total = progress.get("total_steps")
        workflow = progress.get("workflow_type")
        if isinstance(cur, int) and isinstance(total, int) and total > 0:
            current_subtask = (
                f"Run phase '{phase}' — step {cur}/{total} ({workflow or 'workflow'})"
            )
            next_suggested_step = f"Finish step {cur}/{total}, then move to step {cur + 1}/{total}"
        elif phase:
            current_subtask = f"Run phase '{phase}' ({workflow or 'workflow'})"

    # --- SKILL.state carry-over: don't poison the working set with
    # "please provide the list of services…" style clarification
    # responses. When the model asks a clarification question the
    # state should mark itself as ``awaiting_clarification`` so the
    # next turn knows the previous step did not progress.
    last_completed = _sanitize_last_completed_step(last_assistant)
    awaiting_clarification = _is_clarification_request(last_assistant)

    working_set = {
        "current_objective": last_user[:250],
        "current_subtask": current_subtask,
        "last_completed_step": last_completed[:200],
        "last_completed_step_kind": "clarification" if awaiting_clarification else "answer",
        "next_suggested_step": next_suggested_step,
        "open_loops": [],
        "active_files": sorted(set(active_files)),
        "active_tools": [],
        "recent_blockers": recent_blockers,
    }
    write_memory_file(session_id, "working_set.json", working_set)
    return working_set


# ---------------------------------------------------------------------------
# Clarification-request detection
# ---------------------------------------------------------------------------

# The model sometimes re-asks the same "please provide a list of X"
# clarification 3+ times in a row when its memory of earlier work
# has been lost (SKILL.state carry-over bug).  The detection here is
# intentionally conservative: it only fires for short, interrogative
# assistant turns in the major languages the project ships UI strings
# for.  The working_set stores ``last_completed_step_kind="clarification"``
# so downstream code (and humans reading the JSON) can tell at a
# glance that the previous step did not produce a real answer.
_CLARIFICATION_HINTS: tuple[str, ...] = (
    # English
    "please provide",
    "could you provide",
    "could you share",
    "can you provide",
    "can you share",
    "i need more information",
    "could you clarify",
    "what specifically",
    "which ones",
    # Spanish
    "por favor proporciona",
    "podrías proporcionar",
    # French
    "pourriez-vous fournir",
    "veuillez fournir",
    # German
    "bitte geben sie an",
    "könnten sie angeben",
    # Polish
    "proszę podać",
    # Ukrainian (the language used in the buggy session)
    "будь ласка, надайте",
    "надайте список",
    "надайте мені список",
    "будь-ласка надайте",
    "дайте список",
    "який саме",
    "які саме",
    "уточніть",
    "уточнить",
    "я не можу надати",
    "не можу надати відповідь",
    "які саме сервіси",
    "які саме ви",
)


def _is_clarification_request(text: str) -> bool:
    """``True`` when the assistant turn is a short re-asking of the
    same information rather than a real answer. Used by
    :func:`update_working_set` to tag the last step so the next turn
    doesn't repeat the question."""
    if not text:
        return False
    lowered = text.strip().lower()
    if not lowered:
        return False
    # Only short turns can be a clarification; long assistant answers
    # are never treated as a clarification.
    if len(lowered) > 400:
        return False
    # Either it ends with a question mark (or the Cyrillic
    # equivalent "…") OR it carries a strong clarification phrase
    # that makes it clear no new information was returned. The
    # post-mortem of the last session showed models sometimes drop
    # the trailing "?" when they rephrase, so we accept any short
    # turn that contains one of the canonical hints.
    if any(hint in lowered for hint in _CLARIFICATION_HINTS):
        return True
    return lowered.endswith("?") or lowered.endswith("…") or lowered.endswith("...")


def _sanitize_last_completed_step(text: str) -> str:
    """Return a sanitized version of the assistant's last message for
    inclusion in the working set.  Pure clarification questions are
    replaced with a stable marker so future turns know to re-use the
    prior context (durable facts / transcript) instead of treating the
    empty clarification as a completed step."""
    if _is_clarification_request(text):
        return "(awaiting clarification: re-using prior context)"
    return text or ""


def _extract_fact_candidates(user_message: str) -> list[dict[str, Any]]:
    lower = user_message.lower()
    out: list[dict[str, Any]] = []

    # Response-style hints across languages.
    short_hints = (
        # English
        "short", "brief", "concise", "in short",
        # Spanish
        "corto", "breve", "conciso", "en resumen",
        # French
        "court", "bref", "concis", "en bref",
        # Portuguese
        "curto", "breve", "conciso", "em resumo",
        # German
        "kurz", "knapp", "kurz und bündig",
        # Italian
        "corto", "breve", "conciso", "in breve",
        # Polish
        "krótki", "zwięzły", "krótko",
        # Dutch
        "kort", "beknopt", "kort en bondig",
        # Turkish
        "kısa", "özlü", "kısaca",
        # Vietnamese
        "ngắn", "ngắn gọn", "tóm tắt",
        # Japanese
        "短く", "簡潔", "手短に",
        # Korean
        "짧게", "간결하게", "요약",
        # Chinese
        "简短", "简洁", "简明",
        # Hindi
        "संक्षिप्त", "छोटा", "संक्षेप में",
        # Arabic
        "قصير", "موجز", "باختصار",
        # Ukrainian
        "коротк", "стисл", "корот",
    )
    detailed_hints = (
        # English
        "detail", "long", "thorough", "in depth",
        # Spanish
        "detallado", "extenso", "a fondo", "en detalle",
        # French
        "détaillé", "long", "approfondi", "en détail",
        # Portuguese
        "detalhado", "extenso", "aprofundado", "em detalhe",
        # German
        "ausführlich", "detailliert", "gründlich", "im detail",
        # Italian
        "dettagliato", "approfondito", "in dettaglio",
        # Polish
        "szczegółowy", "dokładny", "szczegółowo",
        # Dutch
        "gedetailleerd", "uitgebreid", "grondig",
        # Turkish
        "detaylı", "ayrıntılı", "kapsamlı",
        # Vietnamese
        "chi tiết", "kỹ lưỡng", "tỉ mỉ",
        # Japanese
        "詳しく", "詳細に", "徹底的に",
        # Korean
        "자세하게", "상세히", "철저히",
        # Chinese
        "详细", "详尽", "彻底",
        # Hindi
        "विस्तृत", "विस्तार से", "गहराई से",
        # Arabic
        "مفصل", "بالتفصيل", "متعمق",
        # Ukrainian
        "детальн", "довг", "докладн",
    )

    if any(h in lower for h in short_hints):
        out.append(
            {
                "fact_type": "user_preference",
                "subject": "assistant",
                "predicate": "response_style",
                "object": "short",
                "confidence": 0.95,
            }
        )
    if any(h in lower for h in detailed_hints):
        out.append(
            {
                "fact_type": "user_preference",
                "subject": "assistant",
                "predicate": "response_style",
                "object": "detailed",
                "confidence": 0.9,
            }
        )

    # Language hints.
    lang_hints = [
        ("english", ("english", "in english", "speak english", "англій")),
        ("spanish", ("español", "in spanish", "speak spanish", "hablar español", "en español")),
        ("french", ("français", "in french", "speak french", "parler français", "en français")),
        ("portuguese", ("português", "in portuguese", "speak portuguese", "em português")),
        ("german", ("deutsch", "in german", "speak german", "auf deutsch")),
        ("italian", ("italiano", "in italian", "speak italian", "in italiano")),
        ("polish", ("polski", "in polish", "speak polish", "po polsku")),
        ("dutch", ("nederlands", "in dutch", "speak dutch", "in het nederlands")),
        ("turkish", ("türkçe", "in turkish", "speak turkish")),
        ("chinese", ("中文", "chinese", "in chinese", "speak chinese")),
        ("japanese", ("日本語", "japanese", "in japanese", "speak japanese")),
        ("korean", ("한국어", "korean", "in korean", "speak korean")),
        ("arabic", ("العربية", "arabic", "in arabic", "speak arabic")),
        ("hindi", ("हिन्दी", "hindi", "in hindi", "speak hindi")),
        ("vietnamese", ("tiếng việt", "vietnamese", "in vietnamese", "speak vietnamese")),
        ("ukrainian", ("ukrainian", "in ukrainian", "speak ukrainian", "україн")),
    ]

    for lang_code, hints in lang_hints:
        if any(h in lower for h in hints):
            out.append(
                {
                    "fact_type": "user_preference",
                    "subject": "assistant",
                    "predicate": "language",
                    "object": lang_code,
                    "confidence": 0.95,
                }
            )
            break  # one language preference per turn is enough

    return out


def _upsert_fact(
    *,
    session_id: str,
    fact: dict[str, Any],
    source_message_id: str | None,
    excerpt: str,
) -> str:
    now = utcnow_iso()
    subject = str(fact["subject"])
    predicate = str(fact["predicate"])
    obj = str(fact["object"])
    confidence = float(fact.get("confidence", 0.75))

    existing = fetch_one(
        """
        SELECT * FROM facts
        WHERE session_id=? AND subject=? AND predicate=? AND object=?
        LIMIT 1
        """,
        (session_id, subject, predicate, obj),
    )

    if existing:
        fact_id = existing["id"]
        existing_conf = float(existing["confidence"])
        bumped = min(1.0, max(existing_conf, confidence) + 0.03)
        execute(
            """
            UPDATE facts
            SET confidence=?, is_durable=1, updated_at=?
            WHERE id=?
            """,
            (bumped, now, fact_id),
        )
    else:
        fact_id = str(uuid.uuid4())
        execute(
            """
            INSERT INTO facts (
              id, session_id, fact_type, subject, predicate, object,
              confidence, source_chunk_id, is_durable, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                fact_id,
                session_id,
                str(fact.get("fact_type", "memory_claim")),
                subject,
                predicate,
                obj,
                confidence,
                None,
                now,
                now,
            ),
        )

    if source_message_id:
        row = fetch_one(
            """
            SELECT id FROM claim_sources
            WHERE session_id=? AND fact_id=? AND source_type='message' AND source_ref_id=?
            LIMIT 1
            """,
            (session_id, fact_id, source_message_id),
        )
        if not row:
            execute(
                """
                INSERT INTO claim_sources (
                  id, session_id, fact_id, source_type, source_ref_id, excerpt, created_at
                ) VALUES (?, ?, ?, 'message', ?, ?, ?)
                """,
                (str(uuid.uuid4()), session_id, fact_id, source_message_id, excerpt[:240], now),
            )

    return fact_id


def maybe_update_durable_facts(
    session_id: str,
    user_message: str,
    source_message_id: str | None = None,
) -> list[dict[str, Any]]:
    existing = read_memory_file(session_id, "durable_facts.json")
    facts = list(existing if isinstance(existing, list) else [])

    new_items = _extract_fact_candidates(user_message)

    for candidate in new_items:
        _upsert_fact(
            session_id=session_id,
            fact=candidate,
            source_message_id=source_message_id,
            excerpt=user_message,
        )
        if not any(
            f.get("subject") == candidate["subject"]
            and f.get("predicate") == candidate["predicate"]
            and f.get("object") == candidate["object"]
            for f in facts
        ):
            facts.append(
                {
                    "subject": candidate["subject"],
                    "predicate": candidate["predicate"],
                    "object": candidate["object"],
                    "confidence": candidate.get("confidence", 0.75),
                }
            )

    write_memory_file(session_id, "durable_facts.json", facts)
    if new_items:
        detect_fact_conflicts(session_id)
    return facts


# ---------------------------------------------------------------------------
# SKILL.state carry-over: when a background ``Run`` finishes, the
# per-step facts discovered by the worker are merged into the session
# ``durable_facts.json`` and ``retrieval_anchors.json`` so the next chat
# turn (which may be in ``"full"`` mode or in a fresh ``SKILL.state``
# session) can quote them instead of re-asking the same question.
# ---------------------------------------------------------------------------

# Capped so a runaway research run can't fill the working set with
# thousands of rows. 50 services / facts is plenty for a single
# short-horizon session.
MAX_DURABLE_FACTS_PER_RUN = 50

# Worker roles whose output is "user-facing research" worth carrying
# over. Skip the verifier and the executor (their output is internal).
_CARRYOVER_ROLES: frozenset[str] = frozenset(
    {
        "researcher",
        "github-researcher",
        "web-researcher",
        "extractor",
        "synthesizer",
        "critic",
    }
)


def _extract_entities_from_text(
    text: str,
    *,
    max_entities: int = 20,
) -> list[dict[str, Any]]:
    """Lightweight, LLM-free entity extractor used to seed
    ``durable_facts.json`` from the assistant's answer in a
    foreground chat turn.

    The previous version of the runtime only ever updated
    ``durable_facts`` from background ``Run`` outputs.  That left
    a gaping hole in SKILL.state mode: when the user asked a
    follow-up "give me the answer with those services", the model
    had no idea what was in the previous turn and asked for the
    list again.  This extractor pulls the most informative lines
    out of the assistant text so the next turn can re-use them.

    The heuristic is intentionally simple and language-agnostic:

    * Lines that start with a bullet (``*``, ``-``, ``•``) or a
      bold heading (``**1.**`` …) are kept as-is.
    * Lines that are mostly uppercase / proper-noun are kept
      (e.g. "GitHub: open-free-llm-api/awesome-freellm-apis").
    * Citation patterns like ``[1] lmspeed.net`` are also kept.
    * Bare URLs (``https://…``, ``site.com``) anywhere in the
      text are extracted, since the model often drops the URL
      into a free-form sentence rather than a bullet.

    Returns a list of ``{"claim": ..., "source": "turn_text",
    "confidence": "medium"}`` dicts, capped at ``max_entities``.
    """
    if not text:
        return []
    extracted: list[dict[str, Any]] = []
    seen: set[str] = set()

    url_re = re.compile(
        r"(?P<url>(?:https?://|www\.)[a-zA-Z0-9._/\-]+[a-zA-Z0-9/\-])"
        r"|(?P<bare>[a-zA-Z0-9\-]+\.(?:com|net|ai|io|org|co|dev|app|ru|ua|de|uk)(?:/[a-zA-Z0-9_\-./?#=&%]+)?)"
    )

    def _add(claim: str) -> None:
        if not claim:
            return
        claim = claim.strip().rstrip(",.;: ")
        if len(claim) < 4 or len(claim) > 280:
            return
        norm = re.sub(r"\s+", " ", claim.lower())
        if norm in seen:
            return
        seen.add(norm)
        extracted.append(
            {
                "claim": claim,
                "source": "turn_text",
                "confidence": "medium",
            }
        )

    # 1) Bullet / heading / citation lines.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"---", "***", "==="}:
            continue
        if line.startswith("#") and len(line) < 4:
            continue

        bullet_match = re.match(r"^[\-•·\*]+\s*(?:\d+\.\s*)?(.+)$", line)
        if bullet_match:
            cand = re.sub(r"\*+", "", bullet_match.group(1)).strip()
            _add(cand)
            continue
        if line.startswith("**"):
            cand = re.sub(r"\*+", "", line).strip()
            _add(cand)
            continue
        cite_match = re.match(r"^\[(\d+)\]\s+(\S+)(?:\s+(.{0,200}))?", line)
        if cite_match:
            num = cite_match.group(1)
            url = cite_match.group(2)
            rest = (cite_match.group(3) or "").strip()
            claim = f"[{num}] {url}" + (f" — {rest}" if rest else "")
            _add(claim)
            continue

        if len(extracted) >= max_entities:
            break

    # 2) Bare URLs anywhere in the text (catches the cases
    #    where the model wrote "Kie.ai (https://kie.ai) is...").
    if len(extracted) < max_entities:
        for m in url_re.finditer(text):
            url = m.group("url") or m.group("bare")
            if not url:
                continue
            _add(url)
            if len(extracted) >= max_entities:
                break

    return extracted


def record_turn_entities(
    session_id: str,
    assistant_text: str,
    *,
    source_message_id: str | None = None,
) -> dict[str, Any]:
    """Append entities found in a foreground ``stream_chat``
    assistant turn to ``durable_facts.json``.  This is the
    SKILL.state carry-over equivalent of
    :func:`record_durable_facts_from_run` for the interactive
    chat path.  Returns a small summary for logging.
    """
    if not assistant_text or len(assistant_text) < 30:
        # Skip empty / very short turns (e.g. clarification
        # acks).  Otherwise we would just fill durable_facts with
        # noise.
        return {"added": 0, "skipped_reason": "too_short"}

    # Don't extract from clarifications.
    if _is_clarification_request(assistant_text):
        return {"added": 0, "skipped_reason": "clarification"}

    extracted = _extract_entities_from_text(assistant_text)
    if not extracted:
        return {"added": 0, "skipped_reason": "no_entities"}

    existing = read_memory_file(session_id, "durable_facts.json")
    facts: list[dict[str, Any]] = list(existing if isinstance(existing, list) else [])

    added = 0
    for fact in extracted:
        norm_claim = re.sub(r"\s+", " ", fact["claim"].strip().lower())
        if any(
            re.sub(r"\s+", " ", str(f.get("claim") or "").strip().lower()) == norm_claim
            for f in facts
        ):
            continue
        fact_with_meta = dict(fact)
        if source_message_id:
            fact_with_meta["source_message_id"] = source_message_id
        facts.append(fact_with_meta)
        added += 1

    if added:
        write_memory_file(session_id, "durable_facts.json", facts[-256:])
    return {"added": added, "total": len(facts)}


def _extract_bullet_facts(text: str) -> list[dict[str, Any]]:
    """Pull ``- …`` and ``* …`` style bullets out of a step output.

    These are the most stable form of "service names / facts" the
    research / synthesizer roles produce, and they're cheap to match
    without an LLM call.  We deliberately ignore numbered lists and
    plain prose so the carry-over stays compact."""
    if not text:
        return []
    facts: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Match `*`, `-`, `•`, or `·` followed by content.
        if not re.match(r"^[\-•·]\s+\S", stripped) and not stripped.startswith("**"):
            continue
        # Strip the bullet / bold markers.
        cleaned = re.sub(r"^[\-•·]\s+", "", stripped)
        cleaned = re.sub(r"^\*+", "", cleaned)
        cleaned = re.sub(r"\*+$", "", cleaned)
        cleaned = re.sub(r"\*\*", "", cleaned).strip()
        if len(cleaned) < 8 or len(cleaned) > 320:
            continue
        # Skip noise.
        if cleaned.lower().startswith((
            "url:", "source:", "next step", "next:", "artifacts",
            "criteria", "constraint",
        )):
            continue
        facts.append({"claim": cleaned, "source": "step_output", "confidence": "medium"})
        if len(facts) >= MAX_DURABLE_FACTS_PER_RUN:
            break
    return facts


def record_durable_facts_from_run(
    session_id: str,
    run_id: str,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge the per-step facts discovered by a background ``Run`` into
    the session's persistent memory.

    The original SKILL.state paper (arXiv:2608.26263) drops
    intermediate reasoning as soon as the state is updated. The
    runtime in this project keeps the transcript (so the user can read
    the steps in the chat) but the durable_facts layer was previously
    only populated from explicit ``maybe_update_durable_facts`` calls,
    which only fire on user messages. As a result the second
    background ``Run`` in a long session could not see the facts the
    first one found.  This function closes that gap.

    Returns a small summary dict the caller can log.
    """
    existing = read_memory_file(session_id, "durable_facts.json")
    facts: list[dict[str, Any]] = list(existing if isinstance(existing, list) else [])

    anchors_raw = read_memory_file(session_id, "retrieval_anchors.json")
    anchors: list[dict[str, Any]] = list(anchors_raw if isinstance(anchors_raw, list) else [])

    added_facts = 0
    added_anchors = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        worker_type = str(step.get("worker_type") or "").strip().lower()
        if worker_type not in _CARRYOVER_ROLES:
            continue
        step_output = str(step.get("output") or "")
        bullet_facts = _extract_bullet_facts(step_output)
        for fact in bullet_facts:
            # Dedup against the existing list using a normalised
            # comparison so the file does not bloat when the same
            # service / fact is mentioned twice.
            normalized = re.sub(r"\s+", " ", fact["claim"].strip().lower())
            if any(
                re.sub(r"\s+", " ", str(f.get("claim") or "").strip().lower()) == normalized
                for f in facts
            ):
                continue
            facts.append(fact)
            added_facts += 1
            if len(facts) >= 256:  # hard cap on the per-session list
                break

        # Pin the most recent researcher / synthesizer message as a
        # retrieval anchor so future turns can re-include it in the
        # recall pack even after the chat history is dropped (SKILL.state
        # carry-over across windows).
        result_message_id = step.get("result_message_id")
        if isinstance(result_message_id, str) and result_message_id:
            if not any(a.get("message_id") == result_message_id for a in anchors):
                anchors.append(
                    {
                        "message_id": result_message_id,
                        "kind": "run_step",
                        "worker_type": worker_type,
                        "run_id": run_id,
                    }
                )
                added_anchors += 1

    write_memory_file(session_id, "durable_facts.json", facts[-256:])
    write_memory_file(session_id, "retrieval_anchors.json", anchors[-64:])
    return {
        "added_facts": added_facts,
        "added_anchors": added_anchors,
        "total_facts": len(facts),
        "total_anchors": len(anchors),
    }


def apply_fact_confidence_decay(session_id: str) -> dict[str, Any]:
    rows = fetch_all(
        "SELECT id, confidence, updated_at FROM facts WHERE session_id=?",
        (session_id,),
    )
    now_dt = datetime.now(timezone.utc)
    updated = 0

    for row in rows:
        old_conf = float(row["confidence"])
        updated_at = _parse_iso(row["updated_at"])
        if not updated_at:
            continue
        age_days = max(0.0, (now_dt - updated_at).total_seconds() / 86400.0)
        if age_days <= 0:
            continue
        new_conf = max(FACT_CONFIDENCE_FLOOR, old_conf * ((1.0 - FACT_DECAY_PER_DAY) ** age_days))
        if abs(new_conf - old_conf) < 0.0005:
            continue
        execute(
            "UPDATE facts SET confidence=? WHERE id=?",
            (round(new_conf, 4), row["id"]),
        )
        updated += 1

    return {"facts_checked": len(rows), "facts_updated": updated}


def detect_fact_conflicts(session_id: str) -> list[dict[str, Any]]:
    rows = fetch_all(
        """
        SELECT id, subject, predicate, object, confidence
        FROM facts
        WHERE session_id=? AND is_durable=1 AND confidence >= 0.35
        ORDER BY confidence DESC, updated_at DESC
        """,
        (session_id,),
    )
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["subject"], row["predicate"])
        object_key = str(row["object"]).strip().lower()
        if not object_key:
            continue
        grouped.setdefault(key, {})
        if object_key not in grouped[key]:
            grouped[key][object_key] = dict(row)

    now = utcnow_iso()
    seen_pairs: set[tuple[str, str]] = set()

    for (subject, predicate), by_object in grouped.items():
        facts_for_pair = list(by_object.values())
        if len(facts_for_pair) < 2:
            continue

        for a, b in combinations(facts_for_pair, 2):
            fact_a_id, fact_b_id = sorted([a["id"], b["id"]])
            seen_pairs.add((fact_a_id, fact_b_id))
            explanation = (
                f"Conflicting durable claims for {subject}.{predicate}: "
                f"'{a['object']}' vs '{b['object']}'."
            )
            existing = fetch_one(
                """
                SELECT id FROM fact_conflicts
                WHERE session_id=? AND fact_a_id=? AND fact_b_id=?
                LIMIT 1
                """,
                (session_id, fact_a_id, fact_b_id),
            )
            if existing:
                execute(
                    """
                    UPDATE fact_conflicts
                    SET status='open', explanation=?, last_seen_at=?
                    WHERE id=?
                    """,
                    (explanation, now, existing["id"]),
                )
            else:
                execute(
                    """
                    INSERT INTO fact_conflicts (
                      id, session_id, subject, predicate, fact_a_id, fact_b_id,
                      status, explanation, first_detected_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), session_id, subject, predicate, fact_a_id, fact_b_id, explanation, now, now),
                )

    existing_open = fetch_all(
        "SELECT id, fact_a_id, fact_b_id FROM fact_conflicts WHERE session_id=? AND status='open'",
        (session_id,),
    )
    for row in existing_open:
        pair = (row["fact_a_id"], row["fact_b_id"])
        if pair not in seen_pairs:
            execute(
                "UPDATE fact_conflicts SET status='resolved', last_seen_at=? WHERE id=?",
                (now, row["id"]),
            )

    return list_fact_conflicts(session_id)


def _extract_decisions_tasks(session_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _recent_messages(session_id, limit=40)
    decisions = []
    tasks = []
    now = utcnow_iso()

    for row in rows:
        text = row["content_text"]
        lowered = text.lower()
        if (
    "decided" in lowered
    or "decision" in lowered
    or "we will" in lowered
    # Spanish/French/etc.
    or "decidido" in lowered
    or "decidimos" in lowered
    or "décidé" in lowered
    or "decided" in lowered
    or "entschieden" in lowered
    or "abbiamo deciso" in lowered
    or "zdecydowaliśmy" in lowered
    or "besloten" in lowered
    or "karar verdik" in lowered
    or "決定" in lowered
    or "결정" in lowered
    or "हमने तय किया" in lowered
    or "قررنا" in lowered
    or "вирішили" in lowered
):
            decisions.append(
                {
                    "id": str(uuid.uuid4()),
                    "title": text[:80],
                    "decision_text": text,
                    "rationale": "Extracted from recent conversation",
                    "status": "active",
                    "source_chunk_id": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        if (
    "todo" in lowered
    or "next" in lowered
    or "need to" in lowered
    or "we should" in lowered
    # Spanish/French/etc.
    or "pendiente" in lowered
    or "necesito" in lowered
    or "hay que" in lowered
    or "à faire" in lowered
    or "preciso" in lowered
    or "müssen" in lowered
    or "dovremmo" in lowered
    or "musimy" in lowered
    or "moeten" in lowered
    or "yapmalıyız" in lowered
    or "やること" in lowered
    or "해야 할 일" in lowered
    or "我们需要" in lowered
    or "हमें करना" in lowered
    or "يجب أن" in lowered
    or "потрібно" in lowered
):
            tasks.append(
                {
                    "id": str(uuid.uuid4()),
                    "title": text[:80],
                    "description": text,
                    "status": "open",
                    "priority": 2,
                    "source_chunk_id": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    # Upsert simplified: append new rows.
    for d in decisions:
        execute(
            """
            INSERT INTO decisions (id, session_id, title, decision_text, rationale, status, source_chunk_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                d["id"],
                session_id,
                d["title"],
                d["decision_text"],
                d["rationale"],
                d["status"],
                d["source_chunk_id"],
                d["created_at"],
                d["updated_at"],
            ),
        )

    for t in tasks:
        execute(
            """
            INSERT INTO tasks (id, session_id, title, description, status, priority, source_chunk_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                t["id"],
                session_id,
                t["title"],
                t["description"],
                t["status"],
                t["priority"],
                t["source_chunk_id"],
                t["created_at"],
                t["updated_at"],
            ),
        )

    return decisions, tasks


def create_checkpoint(session_id: str, source_window_id: str, reason: str = "rollover") -> dict[str, Any]:
    row = fetch_one(
        "SELECT COALESCE(MAX(checkpoint_index), 0) AS max_idx FROM checkpoints WHERE session_id=?",
        (session_id,),
    )
    checkpoint_index = int(row["max_idx"]) + 1 if row else 1

    working_set = update_working_set(session_id)
    decisions, _tasks = _extract_decisions_tasks(session_id)

    recent = _recent_messages(session_id, limit=12)
    files_touched = []
    anchors = []
    for msg in recent:
        if msg["is_anchor"]:
            anchors.append({"message_id": msg["id"], "text": msg["content_text"][:180]})
        if msg["message_type"] in {"tool_result", "internal_event"} and "file" in msg["content_text"].lower():
            files_touched.append(msg["content_text"][:120])

    summary_text = "\n".join(
        [
            f"Reason: {reason}",
            f"Current objective: {working_set['current_objective']}",
            f"Last completed: {working_set['last_completed_step']}",
            f"Next step: {working_set['next_suggested_step']}",
        ]
    )

    lint_run = run_wiki_lint(session_id, reason=f"checkpoint:{reason}")

    payload = {
        "summary": summary_text,
        "current_goal": working_set["current_objective"],
        "working_set": {
            "current_subtask": working_set["current_subtask"],
            "last_completed_step": working_set["last_completed_step"],
            "next_step": working_set["next_suggested_step"],
            "open_loops": working_set["open_loops"],
        },
        "decisions_made": decisions[:10],
        "open_questions": [],
        "important_constraints": [],
        "artifacts_created": [],
        "files_touched": files_touched,
        "retrieval_anchors": anchors,
        "lint_summary": lint_run["summary"],
    }

    checkpoint_id = str(uuid.uuid4())
    now = utcnow_iso()

    execute(
        """
        INSERT INTO checkpoints (
          id, session_id, source_window_id, checkpoint_index, created_at, summary_text,
          working_set_json, decisions_json, open_questions_json, constraints_json,
          artifacts_json, files_touched_json, retrieval_anchors_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            checkpoint_id,
            session_id,
            source_window_id,
            checkpoint_index,
            now,
            summary_text,
            json.dumps(payload["working_set"], ensure_ascii=False),
            json.dumps(payload["decisions_made"], ensure_ascii=False),
            json.dumps(payload["open_questions"], ensure_ascii=False),
            json.dumps(payload["important_constraints"], ensure_ascii=False),
            json.dumps(payload["artifacts_created"], ensure_ascii=False),
            json.dumps(payload["files_touched"], ensure_ascii=False),
            json.dumps(payload["retrieval_anchors"], ensure_ascii=False),
        ),
    )

    write_checkpoint_file(session_id, checkpoint_index, payload)
    append_transcript_event(
        session_id,
        {
            "timestamp": now,
            "event": "checkpoint_created",
            "checkpoint_id": checkpoint_id,
            "window_id": source_window_id,
            "checkpoint_index": checkpoint_index,
        },
    )
    return {"id": checkpoint_id, "checkpoint_index": checkpoint_index, "summary": summary_text}


def list_checkpoints(session_id: str) -> list[dict[str, Any]]:
    rows = fetch_all("SELECT * FROM checkpoints WHERE session_id=? ORDER BY checkpoint_index DESC", (session_id,))
    out = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "source_window_id": row["source_window_id"],
                "checkpoint_index": row["checkpoint_index"],
                "created_at": row["created_at"],
                "summary_text": row["summary_text"],
                "working_set_json": json.loads(row["working_set_json"]),
                "decisions_json": json.loads(row["decisions_json"]),
                "open_questions_json": json.loads(row["open_questions_json"]),
                "constraints_json": json.loads(row["constraints_json"]),
                "artifacts_json": json.loads(row["artifacts_json"]),
                "files_touched_json": json.loads(row["files_touched_json"]),
                "retrieval_anchors_json": json.loads(row["retrieval_anchors_json"]),
            }
        )
    return out


def list_memory_table(session_id: str, table_name: str) -> list[dict[str, Any]]:
    rows = fetch_all(f"SELECT * FROM {table_name} WHERE session_id=? ORDER BY created_at DESC", (session_id,))
    return [dict(r) for r in rows]


def list_retrieval_logs(session_id: str) -> list[dict[str, Any]]:
    rows = fetch_all("SELECT * FROM retrieval_logs WHERE session_id=? ORDER BY created_at DESC LIMIT 100", (session_id,))
    result = []
    for row in rows:
        item = dict(row)
        for key in ("filters_json", "results_json", "reranked_results_json", "final_pack_json"):
            item[key] = json.loads(item[key])
        result.append(item)
    return result


def list_claim_sources(session_id: str) -> list[dict[str, Any]]:
    rows = fetch_all(
        """
        SELECT
          cs.*,
          f.subject,
          f.predicate,
          f.object
        FROM claim_sources cs
        LEFT JOIN facts f ON f.id=cs.fact_id
        WHERE cs.session_id=?
        ORDER BY cs.created_at DESC
        LIMIT 300
        """,
        (session_id,),
    )
    return [dict(r) for r in rows]


def list_fact_conflicts(session_id: str) -> list[dict[str, Any]]:
    rows = fetch_all(
        """
        SELECT
          fc.*,
          fa.object AS object_a,
          fa.confidence AS confidence_a,
          fb.object AS object_b,
          fb.confidence AS confidence_b
        FROM fact_conflicts fc
        LEFT JOIN facts fa ON fa.id=fc.fact_a_id
        LEFT JOIN facts fb ON fb.id=fc.fact_b_id
        WHERE fc.session_id=?
        ORDER BY
          CASE WHEN fc.status='open' THEN 0 ELSE 1 END,
          fc.last_seen_at DESC
        LIMIT 200
        """,
        (session_id,),
    )
    return [dict(r) for r in rows]


def list_lint_runs(session_id: str, limit: int = 30) -> list[dict[str, Any]]:
    rows = fetch_all(
        "SELECT * FROM lint_runs WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
        (session_id, limit),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["summary_json"] = json.loads(item["summary_json"])
        out.append(item)
    return out


def run_wiki_lint(session_id: str, reason: str = "manual") -> dict[str, Any]:
    decay = apply_fact_confidence_decay(session_id)
    conflicts = detect_fact_conflicts(session_id)

    anchors_raw = read_memory_file(session_id, "retrieval_anchors.json")
    anchors = anchors_raw if isinstance(anchors_raw, list) else []
    orphan_anchor_ids: list[str] = []
    for anchor in anchors:
        message_id = str(anchor.get("message_id") or "").strip()
        if not message_id:
            continue
        msg = fetch_one("SELECT id FROM messages WHERE session_id=? AND id=? LIMIT 1", (session_id, message_id))
        if msg is None:
            orphan_anchor_ids.append(message_id)

    stale_row = fetch_one(
        "SELECT COUNT(*) AS c FROM facts WHERE session_id=? AND is_durable=1 AND confidence < 0.4",
        (session_id,),
    )
    stale_claims = int(stale_row["c"]) if stale_row else 0

    summary = {
        "open_conflicts": len([c for c in conflicts if c.get("status") == "open"]),
        "resolved_conflicts": len([c for c in conflicts if c.get("status") == "resolved"]),
        "stale_claims": stale_claims,
        "orphan_anchors": len(orphan_anchor_ids),
        "orphan_anchor_ids": orphan_anchor_ids[:20],
        "decay": decay,
    }

    run_id = str(uuid.uuid4())
    now = utcnow_iso()
    execute(
        "INSERT INTO lint_runs (id, session_id, timestamp, reason, summary_json) VALUES (?, ?, ?, ?, ?)",
        (run_id, session_id, now, reason, json.dumps(summary, ensure_ascii=False)),
    )
    append_transcript_event(
        session_id,
        {
            "timestamp": now,
            "event": "wiki_lint_run",
            "run_id": run_id,
            "reason": reason,
            "summary": summary,
        },
    )
    return {"id": run_id, "timestamp": now, "reason": reason, "summary": summary}


def maybe_run_scheduled_wiki_lint(session_id: str) -> dict[str, Any] | None:
    session_row = fetch_one("SELECT total_message_count FROM sessions WHERE id=? LIMIT 1", (session_id,))
    if not session_row:
        return None
    total_message_count = int(session_row["total_message_count"])
    if total_message_count < SCHEDULED_LINT_INTERVAL_MESSAGES:
        return None
    if total_message_count % SCHEDULED_LINT_INTERVAL_MESSAGES != 0:
        return None

    last_run = fetch_one("SELECT timestamp FROM lint_runs WHERE session_id=? ORDER BY timestamp DESC LIMIT 1", (session_id,))
    last_ts = _parse_iso(last_run["timestamp"]) if last_run else None
    if last_ts:
        elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds()
        if elapsed < SCHEDULED_LINT_MIN_SECONDS:
            return None

    return run_wiki_lint(session_id, reason="scheduled_turn")


def pin_message(session_id: str, message_id: str, pinned: bool, anchor: bool) -> None:
    execute(
        "UPDATE messages SET is_pinned=?, is_anchor=? WHERE id=? AND session_id=?",
        (1 if pinned else 0, 1 if anchor else 0, message_id, session_id),
    )

    if anchor:
        anchors = read_memory_file(session_id, "retrieval_anchors.json")
        if not isinstance(anchors, list):
            anchors = []
        if not any(a.get("message_id") == message_id for a in anchors):
            anchors.append({"message_id": message_id})
            write_memory_file(session_id, "retrieval_anchors.json", anchors)


def memory_snapshot(session_id: str) -> dict[str, Any]:
    lint_runs = list_lint_runs(session_id, limit=20)
    return {
        "latest_checkpoint": (list_checkpoints(session_id) or [None])[0],
        "facts": list_memory_table(session_id, "facts"),
        "durable_facts": read_memory_file(session_id, "durable_facts.json"),
        "claim_sources": list_claim_sources(session_id),
        "fact_conflicts": list_fact_conflicts(session_id),
        "decisions": list_memory_table(session_id, "decisions"),
        "tasks": list_memory_table(session_id, "tasks"),
        "anchors": read_memory_file(session_id, "retrieval_anchors.json"),
        "retrieval_logs": list_retrieval_logs(session_id),
        "lint_runs": lint_runs,
        "latest_lint": lint_runs[0] if lint_runs else None,
    }


# ---------------------------------------------------------------------------
# SKILL.state bridge — thin re-exports so the orchestrator only depends on
# `memory` (its existing import surface) and not on the lower-level
# ``skill_state`` module.
# ---------------------------------------------------------------------------


def start_or_resume_skill(
    session_id: str,
    skill_name: str,
    *,
    user_prompt: str | None = None,
) -> dict[str, Any]:
    """Load or create the execution state for ``skill_name`` in this
    session. If ``user_prompt`` is supplied and the state has no
    observations yet it is recorded as the first observation."""
    state = _skill_state_start_or_resume(session_id, skill_name, user_prompt=user_prompt)
    return state.to_dict()


def reset_skill_state(session_id: str, skill_name: str) -> dict[str, Any]:
    state = _skill_state_reset_state(session_id, skill_name)
    return state.to_dict()


def apply_skill_transition(
    session_id: str,
    skill_name: str,
    *,
    transition: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    user_prompt: str | None = None,
) -> dict[str, Any]:
    """Validate a proposed transition against the persisted state and
    persist the new state. The intermediate reasoning produced by the
    model is never written to disk — only validated observations and
    state updates survive."""
    try:
        state = _skill_state_apply_step(
            session_id,
            skill_name,
            transition=transition,
            observation=observation,
            user_prompt=user_prompt,
        )
    except _SkillTransitionError:
        raise
    return state.to_dict()


def load_skill_state(session_id: str, skill_name: str) -> dict[str, Any] | None:
    state = _skill_state_load_state(session_id, skill_name)
    return state.to_dict() if state else None


def list_skill_states(session_id: str) -> list[dict[str, Any]]:
    return _skill_state_list_states(session_id)


def build_skill_prompt_bundle(session_id: str, skill_name: str) -> dict[str, Any]:
    return _skill_state_build_prompt_bundle(session_id, skill_name)


def plan_skill_delegation(
    session_id: str,
    skill_name: str,
    *,
    user_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan the next delegated tool call for a skill with a
    ``delegates_to`` block. Handles ``web-search-loop`` and similar
    multi-engine retry skills by reading the persisted state and
    rotating ``prefer_engine`` when the previous attempt was empty.
    """
    return _skill_state_plan_skill_delegation(session_id, skill_name, user_args=user_args)


def record_skill_tool_observation(
    session_id: str,
    skill_name: str,
    *,
    tool: str,
    result_text: str | None,
    is_empty: bool | None = None,
) -> dict[str, Any]:
    """Push a tool observation onto a skill's history with the
    ``empty_result`` flag. Used by callers that wire a skill's
    ``delegates_to.tool`` to a real MCP call so the next call can
    rotate the engine when needed."""
    state = _skill_state_record_skill_tool_observation(
        session_id,
        skill_name,
        tool=tool,
        result_text=result_text,
        is_empty=is_empty,
    )
    return state.to_dict()
