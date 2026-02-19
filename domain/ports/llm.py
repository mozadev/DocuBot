"""Port: contrato para cualquier proveedor de LLM."""

from __future__ import annotations

from typing import Protocol, List, Dict, Any, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    """Interfaz que debe cumplir cualquier LLM (OpenAI, Anthropic, Ollama, etc.)."""

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """Envía mensajes y retorna la respuesta como texto."""
        ...

    def describe_image(self, image_path: str, context: str = "") -> str:
        """Describe una imagen usando capacidades de visión."""
        ...
