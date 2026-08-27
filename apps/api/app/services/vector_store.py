from __future__ import annotations

from typing import Any

from ..config import load_app_config, settings
from ..db import fetch_all
from .embeddings import cosine_similarity, embed_text

try:
    import lancedb  # type: ignore
except Exception:  # pragma: no cover
    lancedb = None


class VectorStore:
    def __init__(self) -> None:
        self.cfg = load_app_config()
        self._db = None
        self._table = None
        if lancedb is not None:
            try:
                self._db = lancedb.connect(settings.lancedb_path)
                self._table = self._db.create_table(
                    "session_chunks",
                    data=[],
                    mode="create_if_not_exists",
                )
            except Exception:
                self._db = None
                self._table = None

    @property
    def dims(self) -> int:
        return self.cfg.embedding_config.dimensions

    def upsert_chunk(
        self,
        chunk_id: str,
        session_id: str,
        window_id: str,
        chunk_type: str,
        text: str,
        created_at: str,
    ) -> list[float]:
        emb = embed_text(text, dimensions=self.dims)
        if self._table is not None:
            row = {
                "chunk_id": chunk_id,
                "session_id": session_id,
                "window_id": window_id,
                "chunk_type": chunk_type,
                "text": text,
                "created_at": created_at,
                "vector": emb,
            }
            try:
                self._table.add([row])
            except Exception:
                pass
        return emb

    def search(
        self,
        session_id: str,
        query_text: str,
        top_k: int,
        metadata_filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        query_vec = embed_text(query_text, dimensions=self.dims)

        if self._table is not None:
            try:
                dataset = self._table.search(query_vec).limit(top_k * 2).to_list()
                results = []
                for item in dataset:
                    if item.get("session_id") != session_id:
                        continue
                    if metadata_filters.get("chunk_type") and item.get("chunk_type") != metadata_filters["chunk_type"]:
                        continue
                    results.append(
                        {
                            "chunk_id": item.get("chunk_id"),
                            "score": float(item.get("_distance", 0.0)),
                            "source": "vector",
                        }
                    )
                results.sort(key=lambda x: x["score"], reverse=False)
                # LanceDB distance is lower-is-better; normalize to higher-is-better.
                normalized = []
                for idx, r in enumerate(results[: top_k * 2]):
                    normalized.append(
                        {
                            **r,
                            "score": 1.0 / (1.0 + max(r["score"], 0.0)),
                            "rank": idx + 1,
                        }
                    )
                return normalized[:top_k]
            except Exception:
                pass

        rows = fetch_all(
            "SELECT id, text, chunk_type, session_id FROM chunks WHERE session_id=?",
            (session_id,),
        )
        scored = []
        for row in rows:
            if metadata_filters.get("chunk_type") and row["chunk_type"] != metadata_filters["chunk_type"]:
                continue
            score = cosine_similarity(query_vec, embed_text(row["text"], dimensions=self.dims))
            scored.append({"chunk_id": row["id"], "score": score, "source": "vector"})
        scored.sort(key=lambda x: x["score"], reverse=True)
        for idx, item in enumerate(scored):
            item["rank"] = idx + 1
        return scored[:top_k]


vector_store = VectorStore()
