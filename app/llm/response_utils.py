from __future__ import annotations

from typing import Any


def coerce_llm_text(response: Any) -> str:
    if response is None:
        return ""

    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            maybe_text = getattr(item, "text", None)
            if isinstance(maybe_text, str):
                parts.append(maybe_text)
        if parts:
            return "\n".join(part for part in parts if part)

    if isinstance(response, dict):
        if isinstance(response.get("text"), str):
            return response["text"]
        if isinstance(response.get("content"), str):
            return response["content"]

    return str(response)
