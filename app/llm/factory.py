"""
llm/factory.py
Unified LLM factory supporting Anthropic Claude, OpenAI, and Google Gemini.

This version keeps the same public API, but nudges the LLM toward more reliable
JSON generation by:
- keeping temperature low for structured tasks
- setting a larger output budget
- requesting application/json when supported
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from app.config import CONFIG
from app.logging_config import get_logger

logger = get_logger(__name__)


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


_DEFAULT_MODELS: dict[str, str] = {
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.GEMINI: "gemini-2.0-flash",
}

_PROVIDER_LABELS: dict[str, str] = {
    LLMProvider.ANTHROPIC: "Anthropic Claude",
    LLMProvider.OPENAI: "OpenAI",
    LLMProvider.GEMINI: "Google Gemini",
}


class LLMConfigError(RuntimeError):
    """Raised when an LLM cannot be configured."""


def build_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> Any:
    resolved_provider = (provider or CONFIG.llm_provider).lower().strip()
    resolved_model = model or CONFIG.llm_model_name or _DEFAULT_MODELS.get(resolved_provider, "")
    resolved_temp = temperature if temperature is not None else CONFIG.llm_temperature
    resolved_tokens = max_tokens if max_tokens is not None else CONFIG.llm_max_tokens

    label = _PROVIDER_LABELS.get(resolved_provider, resolved_provider)
    logger.info("Building LLM: provider=%s model=%s", label, resolved_model)

    if resolved_provider == LLMProvider.ANTHROPIC:
        return _build_anthropic(resolved_model, resolved_temp, resolved_tokens, **kwargs)
    if resolved_provider == LLMProvider.OPENAI:
        return _build_openai(resolved_model, resolved_temp, resolved_tokens, **kwargs)
    if resolved_provider == LLMProvider.GEMINI:
        return _build_gemini(resolved_model, resolved_temp, resolved_tokens, **kwargs)

    raise LLMConfigError(
        f"Unknown LLM provider '{resolved_provider}'. "
        f"Supported: {', '.join(p.value for p in LLMProvider)}"
    )


def build_llm_with_fallback(
    providers: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    order = providers or [LLMProvider.ANTHROPIC, LLMProvider.OPENAI, LLMProvider.GEMINI]
    errors: list[str] = []

    for provider_name in order:
        try:
            return build_llm(provider=provider_name, **kwargs)
        except LLMConfigError as exc:
            errors.append(f"{provider_name}: {exc}")

    raise LLMConfigError("No LLM provider could be configured. Tried:\n" + "\n".join(errors))


def detect_available_providers() -> list[str]:
    available: list[str] = []
    if CONFIG.anthropic_api_key:
        available.append(LLMProvider.ANTHROPIC)
    if CONFIG.openai_api_key:
        available.append(LLMProvider.OPENAI)
    if CONFIG.google_api_key:
        available.append(LLMProvider.GEMINI)
    return available


def _build_anthropic(model: str, temperature: float, max_tokens: int, **kwargs: Any) -> Any:
    if not CONFIG.anthropic_api_key:
        raise LLMConfigError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or set LLM_PROVIDER=openai / LLM_PROVIDER=gemini."
        )

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise LLMConfigError(
            "langchain-anthropic is not installed. Run: pip install langchain-anthropic"
        ) from exc

    return ChatAnthropic(
        model=model or _DEFAULT_MODELS[LLMProvider.ANTHROPIC],
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=CONFIG.anthropic_api_key,
        **kwargs,
    )


def _build_openai(model: str, temperature: float, max_tokens: int, **kwargs: Any) -> Any:
    if not CONFIG.openai_api_key:
        raise LLMConfigError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or set LLM_PROVIDER=anthropic / LLM_PROVIDER=gemini."
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LLMConfigError(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        ) from exc

    return ChatOpenAI(
        model=model or _DEFAULT_MODELS[LLMProvider.OPENAI],
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=CONFIG.openai_api_key,
        **kwargs,
    )


def _build_gemini(model: str, temperature: float, max_tokens: int, **kwargs: Any) -> Any:
    if not CONFIG.google_api_key:
        raise LLMConfigError(
            "GOOGLE_API_KEY is not set. "
            "Add it to your .env file or set LLM_PROVIDER=anthropic / LLM_PROVIDER=openai."
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise LLMConfigError(
            "langchain-google-genai is not installed. "
            "Run: pip install langchain-google-genai"
        ) from exc

    gemini_kwargs = dict(kwargs)

    # Safer defaults for structured JSON tasks.
    gemini_kwargs.setdefault("response_mime_type", "application/json")
    gemini_kwargs.setdefault("max_retries", 2)

    return ChatGoogleGenerativeAI(
        model=model or _DEFAULT_MODELS[LLMProvider.GEMINI],
        temperature=temperature,
        max_output_tokens=max(max_tokens, 2048),
        google_api_key=CONFIG.google_api_key,
        **gemini_kwargs,
    )


__all__ = [
    "LLMProvider",
    "LLMConfigError",
    "build_llm",
    "build_llm_with_fallback",
    "detect_available_providers",
]