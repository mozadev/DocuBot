"""Adapter: LLMPort backed by OpenAI (chat + vision)."""

from __future__ import annotations

import base64
from pathlib import Path

from langchain_openai import ChatOpenAI
from openai import OpenAI

from core.logger import logger

VISION_SYSTEM_PROMPT = (
    "You describe figures extracted from documents so they can be retrieved by "
    "semantic search. Be specific and factual. For a chart, state the axes, the "
    "series and the trend, and read off notable values. For a table, transcribe "
    "the data. For a diagram, name the components and how they connect. Do not "
    "speculate about anything the image does not show."
)


class OpenAIAdapter:
    """
    OpenAI chat and vision.

    Two models rather than one: gpt-4o-mini answers questions, and a stronger
    vision model reads figures at ingest time. Ingest happens once per document
    while answering happens on every turn, so paying more for accuracy at ingest
    is the cheap side of the trade.
    """

    def __init__(
        self, api_key: str, model: str, vision_model: str, temperature: float = 0.2
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._vision_model = vision_model
        self._temperature = temperature
        self._api_key = api_key

    def invoke(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=self._temperature
        )
        return (response.choices[0].message.content or "").strip()

    def describe_image(self, image_path: str, context: str = "") -> str:
        """Return a searchable description of an image, or a marker on failure."""
        try:
            image_data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
            ext = Path(image_path).suffix.lstrip(".").lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
                ext, "image/png"
            )

            response = self._client.chat.completions.create(
                model=self._vision_model,
                messages=[
                    {"role": "system", "content": VISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this figure."
                                + (f" Context: {context}" if context else ""),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{image_data}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=1000,
                temperature=0.2,
            )
            logger.info("Described figure: %s", Path(image_path).name)
            return (response.choices[0].message.content or "").strip()

        except Exception as e:  # noqa: BLE001 - one bad figure must not fail the upload
            logger.error("Vision description failed for %s: %s", image_path, e)
            return f"[image not processed: {Path(image_path).name}]"

    def get_langchain_llm(self) -> ChatOpenAI:
        """A ChatOpenAI instance for use inside the LangGraph agent."""
        return ChatOpenAI(
            model=self._model, temperature=self._temperature, api_key=self._api_key
        )
