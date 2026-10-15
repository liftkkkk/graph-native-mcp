"""Embedding interface with an offline deterministic fallback.

Three backends (mirroring gx-memory's GX_EMBEDDING_BACKEND):
  - "hash"  (default): feature-hashing embeddings. Zero dependencies, zero
    network, fully deterministic. Good for tests and demos; semantic quality
    is limited (it matches surface terms, not meaning).
  - "openai": any OpenAI-compatible endpoint (OpenAI, LM Studio, vLLM,
    Ollama's /v1, ...). Configure via GNM_EMBED_BASE_URL / _API_KEY / _MODEL.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request

HASH_DIM = 256


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


class HashingEmbedder:
    """Deterministic feature-hashing bag-of-words embedding."""

    name = "hash"
    dim = HASH_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * HASH_DIM
            toks = _tokens(t)
            for tok in toks:
                h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "big")
                v[h % HASH_DIM] += 1.0
            # also hash character 2-grams to give CJK text some robustness
            for i in range(len(t) - 1):
                g = t[i:i + 2]
                h = int.from_bytes(hashlib.md5(g.encode("utf-8")).digest()[:4], "big")
                v[h % HASH_DIM] += 0.5
            out.append(v)
        return out


class OpenAICompatEmbedder:
    """OpenAI-compatible /v1/embeddings endpoint."""

    name = "openai"
    dim = None  # learned from first response

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.dim = len(data["data"][0]["embedding"])
        return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]


def get_embedding_provider():
    """Reads GNM_EMBED_* env vars; falls back to the offline hash embedder."""
    backend = os.environ.get("GNM_EMBED_BACKEND", "hash").lower()
    if backend == "openai":
        base_url = os.environ.get("GNM_EMBED_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("GNM_EMBED_API_KEY", "")
        model = os.environ.get("GNM_EMBED_MODEL", "text-embedding-3-small")
        if api_key:
            return OpenAICompatEmbedder(base_url, api_key, model)
    return HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
