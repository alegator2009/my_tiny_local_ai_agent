from __future__ import annotations

import json
import uuid
from typing import Any

from ..config import load_app_config
from ..db import execute, fetch_all, utcnow_iso
from .indexing import recent_neighbor_chunks
from .vector_store import vector_store


def keyword_search(session_id: str, query: str, top_k: int, filters: dict[str, Any]) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    params: list[Any] = [session_id, query, top_k * 2]
    sql = (
        """
        SELECT c.id AS chunk_id, bm25(chunks_fts) AS rank, c.chunk_type, c.window_id
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.chunk_id
        WHERE chunks_fts.session_id = ?
          AND chunks_fts MATCH ?
        """
    )

    if filters.get("chunk_type"):
        sql += " AND c.chunk_type = ?"
        params.insert(-1, filters["chunk_type"])
    if filters.get("window_id"):
        sql += " AND c.window_id = ?"
        params.insert(-1, filters["window_id"])

    sql += " ORDER BY rank LIMIT ?"

    try:
        rows = fetch_all(sql, tuple(params))
    except Exception:
        return []
    out = []
    for idx, row in enumerate(rows):
        # bm25 lower is better.
        score = 1.0 / (1.0 + abs(float(row["rank"])))
        out.append({"chunk_id": row["chunk_id"], "score": score, "source": "keyword", "rank": idx + 1})
    return out[:top_k]


def _fuse(keyword: list[dict[str, Any]], semantic: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_chunk: dict[str, dict[str, Any]] = {}
    for item in keyword:
        by_chunk.setdefault(item["chunk_id"], {"chunk_id": item["chunk_id"], "keyword": 0.0, "semantic": 0.0})
        by_chunk[item["chunk_id"]]["keyword"] = max(by_chunk[item["chunk_id"]]["keyword"], item["score"])
    for item in semantic:
        by_chunk.setdefault(item["chunk_id"], {"chunk_id": item["chunk_id"], "keyword": 0.0, "semantic": 0.0})
        by_chunk[item["chunk_id"]]["semantic"] = max(by_chunk[item["chunk_id"]]["semantic"], item["score"])

    fused = []
    for value in by_chunk.values():
        score = 0.45 * value["keyword"] + 0.55 * value["semantic"]
        fused.append({"chunk_id": value["chunk_id"], "score": score})
    fused.sort(key=lambda x: x["score"], reverse=True)
    for idx, item in enumerate(fused):
        item["rank"] = idx + 1
    return fused


def _cheap_rerank(session_id: str, fused: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not fused:
        return []
    chunk_ids = [x["chunk_id"] for x in fused]
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = fetch_all(
        f"SELECT id, recency_score, importance_score FROM chunks WHERE session_id=? AND id IN ({placeholders})",
        (session_id, *chunk_ids),
    )
    bonuses = {r["id"]: (float(r["recency_score"]), float(r["importance_score"])) for r in rows}

    reranked = []
    for item in fused:
        recency, importance = bonuses.get(item["chunk_id"], (0.0, 0.0))
        score = item["score"] + 0.05 * recency + 0.1 * importance
        reranked.append({**item, "score": score})
    reranked.sort(key=lambda x: x["score"], reverse=True)
    for idx, item in enumerate(reranked):
        item["rank"] = idx + 1
    return reranked


def run_retrieval(
    *,
    session_id: str,
    window_id: str,
    trigger_reason: str,
    query: str,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    filters = filters or {}
    cfg = load_app_config()
    top_k = cfg.retrieval_config.top_k

    kw = keyword_search(session_id, query, top_k=top_k, filters=filters)
    sem = vector_store.search(session_id, query_text=query, top_k=top_k, metadata_filters=filters)

    if cfg.retrieval_config.mode == "keyword_only":
        fused = [{"chunk_id": x["chunk_id"], "score": x["score"], "rank": x["rank"]} for x in kw]
    elif cfg.retrieval_config.mode == "semantic_only":
        fused = [{"chunk_id": x["chunk_id"], "score": x["score"], "rank": x["rank"]} for x in sem]
    else:
        fused = _fuse(kw, sem)

    reranked = _cheap_rerank(session_id, fused) if cfg.retrieval_config.rerank_mode != "off" else fused
    top = reranked[:top_k]

    expanded = recent_neighbor_chunks(
        session_id,
        [x["chunk_id"] for x in top],
        prev_count=cfg.retrieval_config.neighbor_prev,
        next_count=cfg.retrieval_config.neighbor_next,
    )

    ids = [c["id"] for c in expanded][: top_k * 3]
    if ids:
        placeholders = ",".join("?" for _ in ids)
        rows = fetch_all(
            f"SELECT id, chunk_type, text, start_message_id, end_message_id FROM chunks WHERE id IN ({placeholders})",
            tuple(ids),
        )
    else:
        rows = []

    extracts = []
    for row in rows[: max(2, top_k)]:
        extracts.append(
            {
                "chunk_id": row["id"],
                "chunk_type": row["chunk_type"],
                "extract": row["text"][:350],
                "origin": {
                    "start_message_id": row["start_message_id"],
                    "end_message_id": row["end_message_id"],
                },
            }
        )

    decision_rows = fetch_all(
        "SELECT id, title, decision_text FROM decisions WHERE session_id=? ORDER BY updated_at DESC LIMIT 5",
        (session_id,),
    )
    facts_rows = fetch_all(
        "SELECT id, subject, predicate, object FROM facts WHERE session_id=? ORDER BY updated_at DESC LIMIT 5",
        (session_id,),
    )

    recall_pack = {
        "reason_for_retrieval": trigger_reason,
        "facts_or_decisions": {
            "decisions": [dict(r) for r in decision_rows],
            "facts": [dict(r) for r in facts_rows],
        },
        "chunk_extracts": extracts,
        "origin_ids": [x["chunk_id"] for x in top],
        "synthesis": "Use extracted chunks as supporting memory; prioritize explicit latest user instructions.",
    }

    log_id = str(uuid.uuid4())
    now = utcnow_iso()
    execute(
        """
        INSERT INTO retrieval_logs (
          id, session_id, window_id, trigger_reason, query_text, query_type,
          filters_json, results_json, reranked_results_json, final_pack_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            log_id,
            session_id,
            window_id,
            trigger_reason,
            query,
            cfg.retrieval_config.mode,
            json.dumps(filters, ensure_ascii=False),
            json.dumps({"keyword": kw, "semantic": sem}, ensure_ascii=False),
            json.dumps(reranked, ensure_ascii=False),
            json.dumps(recall_pack, ensure_ascii=False),
            now,
        ),
    )

    return {
        "log_id": log_id,
        "keyword_results": kw,
        "semantic_results": sem,
        "reranked_results": reranked,
        "recall_pack": recall_pack,
    }


def should_trigger_retrieval(user_message: str) -> bool:
    triggers = [
        # English
        "earlier", "before", "previously", "remind", "find", "recall",
        "what did we", "we discussed", "we decided",
        # Spanish
        "antes", "previamente", "recuérdame", "encuentra",
        "qué discutimos", "lo que hablamos",
        # French
        "auparavant", "avant", "rappelle-moi", "retrouve",
        "de quoi avons-nous discuté",
        # Portuguese
        "antes", "anteriormente", "lembre-me", "encontre",
        "o que discutimos",
        # German
        "früher", "zuvor", "erinnere mich", "finde",
        "was haben wir besprochen",
        # Italian
        "prima", "precedentemente", "ricordami", "trova",
        "cosa abbiamo discusso",
        # Polish
        "wcześniej", "poprzednio", "przypomnij", "znajdź",
        "o czym rozmawialiśmy",
        # Dutch
        "eerder", "voorheen", "herinner me", "vind",
        "wat bespraken we",
        # Turkish
        "önce", "daha önce", "hatırlat", "bul",
        "ne konuştuk",
        # Vietnamese
        "trước đó", "trước", "nhắc lại", "tìm",
        "chúng ta đã thảo luận",
        # Japanese
        "以前", "前に", "思い出して", "見つけて",
        "以前話した",
        # Korean
        "이전에", "예전에", "기억나", "찾아봐",
        "우리가 이야기한",
        # Chinese
        "之前", "以前", "提醒我", "找一下",
        "我们讨论过",
        # Hindi
        "पहले", "पूर्व में", "याद दिलाओ", "खोजो",
        "हमने चर्चा की थी",
        # Arabic
        "سابقا", "قبل", "ذكرني", "ابحث عن",
        "ما ناقشناه",
        # Ukrainian
        "раніше", "ти вже казав", "повернись",
        "що ми вирішили", "знайди", "нагадай",
    ]
    lowered = user_message.lower()
    return any(t in lowered for t in triggers)
