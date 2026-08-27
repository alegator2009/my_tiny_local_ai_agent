from __future__ import annotations

import hashlib
import math
from typing import Iterable


def embed_text(text: str, dimensions: int = 128) -> list[float]:
    if not text:
        return [0.0] * dimensions
    vec = [0.0] * dimensions
    words = text.lower().split()
    for idx, token in enumerate(words):
        digest = hashlib.sha256(f"{idx}:{token}".encode("utf-8")).digest()
        for i in range(dimensions):
            vec[i] += digest[i % len(digest)] / 255.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list) or not a_list:
        return 0.0
    dot = sum(x * y for x, y in zip(a_list, b_list))
    na = math.sqrt(sum(x * x for x in a_list)) or 1.0
    nb = math.sqrt(sum(y * y for y in b_list)) or 1.0
    return dot / (na * nb)
