"""Adapter: implementación de LLMPort con OpenAI."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from langchain_openai import ChatOpenAI

from core.logger import logger


class OpenAIAdapter:
    """LLM adapter para OpenAI (chat + vision)."""

    def __init__(self, api_key: str, model: str, vision_model: str, temperature: float = 0.2) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._vision_model = vision_model
        self._temperature = temperature
        self._api_key = api_key

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )
        return response.choices[0].message.content.strip()

    def describe_image(self, image_path: str, context: str = "") -> str:
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = Path(image_path).suffix.lstrip(".").lower()
            mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
            mime_type = mime_map.get(ext, "image/png")

            response = self._client.chat.completions.create(
                model=self._vision_model,
                messages=[
                    {"role": "system", "content": (
                        "Eres un experto en análisis de imágenes de documentos. "
                        "Describe de forma detallada y estructurada el contenido. "
                        "Si es gráfico: ejes, valores, tendencias. "
                        "Si es tabla: extrae datos. Si es diagrama: componentes y relaciones. "
                        "Responde en español."
                    )},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Describe esta imagen.{f' Contexto: {context}' if context else ''}"},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}", "detail": "high"}},
                    ]},
                ],
                max_tokens=1000,
                temperature=0.2,
            )
            description = response.choices[0].message.content.strip()
            logger.info(f"Imagen descrita: {Path(image_path).name}")
            return description

        except Exception as e:
            logger.error(f"Error describiendo imagen {image_path}: {e}")
            return f"[Imagen no procesada: {Path(image_path).name}]"

    def get_langchain_llm(self) -> ChatOpenAI:
        """Retorna un ChatOpenAI para usar en cadenas LangChain/LangGraph."""
        return ChatOpenAI(
            model=self._model,
            temperature=self._temperature,
            api_key=self._api_key,
        )
