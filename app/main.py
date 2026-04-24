"""
main.py
Entry point: validate LLM config then launch the Gradio app.
"""

from __future__ import annotations

import sys

from app.config import CONFIG
from app.logging_config import get_logger

logger = get_logger(__name__)


def _check_llm() -> None:
    """Warn (not crash) if no API key is set for the configured provider."""
    from app.llm.factory import detect_available_providers
    available = detect_available_providers()
    if not available:
        logger.warning(
            "No LLM API key detected. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY in .env"
        )
    elif CONFIG.llm_provider not in available:
        logger.warning(
            "LLM_PROVIDER=%s but no key found for it. Available: %s",
            CONFIG.llm_provider, available,
        )
    else:
        logger.info("LLM provider: %s / %s ✓", CONFIG.llm_provider, CONFIG.llm_model_name)


if __name__ == "__main__":
    _check_llm()
    from app.ui.gradio_app import build_ui
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
