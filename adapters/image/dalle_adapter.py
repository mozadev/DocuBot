"""Adapter: generacion de imagenes con DALL-E 3."""

from __future__ import annotations

import os
import base64
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

from openai import OpenAI

from core.logger import logger


class DalleAdapter:
    """Genera imagenes para ads/marketing usando DALL-E 3."""

    def __init__(self, api_key: str, output_dir: str = "./data/generated_images") -> None:
        self._client = OpenAI(api_key=api_key)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DalleAdapter: inicializado (output: {self._output_dir})")

    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        n: int = 1,
    ) -> Dict[str, Any]:
        """Genera una imagen con DALL-E 3.

        Args:
            prompt: Descripcion de la imagen
            size: "1024x1024" (cuadrado), "1792x1024" (landscape), "1024x1792" (portrait)
            quality: "standard" o "hd"
            style: "vivid" (mas colorido) o "natural" (mas fotografico)
        """
        try:
            response = self._client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=n,
                response_format="b64_json",
            )

            results = []
            for i, img_data in enumerate(response.data):
                filename = self._save_image(img_data.b64_json, prompt, i)
                results.append({
                    "filename": filename,
                    "path": str(self._output_dir / filename),
                    "revised_prompt": img_data.revised_prompt or prompt,
                    "size": size,
                    "quality": quality,
                })

            logger.info(f"DALL-E: {len(results)} imagen(es) generada(s)")
            return {"images": results, "error": None}

        except Exception as e:
            logger.error(f"DALL-E error: {e}")
            return {"images": [], "error": str(e)}

    def generate_ad_image(
        self,
        product_description: str,
        brand_colors: list[str] | None = None,
        channel: str = "instagram_feed",
        style: str = "vivid",
    ) -> Dict[str, Any]:
        """Genera una imagen optimizada para ads."""
        size_map = {
            "instagram_feed": "1024x1024",
            "instagram_story": "1024x1792",
            "facebook_feed": "1024x1024",
            "facebook_cover": "1792x1024",
            "ad_landscape": "1792x1024",
            "ad_portrait": "1024x1792",
            "ad_square": "1024x1024",
        }
        size = size_map.get(channel, "1024x1024")

        color_hint = ""
        if brand_colors:
            color_hint = f" Usa estos colores de marca: {', '.join(brand_colors)}."

        enhanced_prompt = (
            f"Imagen publicitaria profesional para redes sociales. "
            f"{product_description}{color_hint} "
            f"Estilo limpio, moderno, alta calidad. "
            f"NO incluir texto ni letras en la imagen. "
            f"Fondo atractivo con buena iluminacion."
        )

        return self.generate(prompt=enhanced_prompt, size=size, style=style)

    def _save_image(self, b64_data: str, prompt: str, index: int) -> str:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        filename = f"dalle_{prompt_hash}_{index}.png"
        path = self._output_dir / filename
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64_data))
        return filename
