"""Port: the contract any LLM provider must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    """Implemented by OpenAIAdapter; Anthropic, Bedrock or Ollama would fit too."""

    def invoke(self, messages: list[dict[str, str]]) -> str:
        """Send messages, return the reply as plain text."""
        ...

    def describe_image(self, image_path: str, context: str = "") -> str:
        """Describe an image for indexing. Returns a marker string on failure."""
        ...

    def get_langchain_llm(self):
        """A LangChain chat model, for use inside the agent graph."""
        ...
