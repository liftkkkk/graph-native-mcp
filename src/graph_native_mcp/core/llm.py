"""Minimal OpenAI-compatible chat client for the graph-build server.

The build server is the ONLY server that calls an LLM. Runtime servers
(graph-rag, graph-agent planning core) stay deterministic: retrieval finds
evidence, the model judges - that division of labor is the point (Ch.5.2.1).

Configure via environment variables:
    GNM_LLM_BASE_URL  e.g. https://api.openai.com/v1 (or LM Studio / vLLM / Ollama)
    GNM_LLM_API_KEY
    GNM_LLM_MODEL     e.g. gpt-4o-mini
"""

from __future__ import annotations

import json
import os
import urllib.request


def llm_configured() -> bool:
    return bool(os.environ.get("GNM_LLM_API_KEY")) and bool(os.environ.get("GNM_LLM_BASE_URL"))


def chat(messages: list[dict], temperature: float = 0.0,
         json_mode: bool = False, max_tokens: int = 2000) -> str:
    base = os.environ["GNM_LLM_BASE_URL"].rstrip("/")
    key = os.environ["GNM_LLM_API_KEY"]
    model = os.environ.get("GNM_LLM_MODEL", "gpt-4o-mini")
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def chat_json(messages: list[dict], retries: int = 1) -> dict:
    """Chat expecting a JSON object; one repair round-trip on parse failure."""
    for attempt in range(retries + 1):
        text = chat(messages, json_mode=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt == retries:
                raise
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "That was not valid JSON. Reply again with a single JSON object only."},
            ]
    raise RuntimeError("unreachable")
