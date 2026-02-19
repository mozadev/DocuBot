"""Tools de creacion: ad copy, imagenes, A/B, video, email."""

from __future__ import annotations
from langchain_core.tools import tool


def build_creation_tools(vector_store, dalle_adapter=None) -> list:

    @tool
    def generate_ad_copy(
        product: str,
        audience: str,
        channel: str,
        tone: str = "profesional",
        objective: str = "conversion",
    ) -> str:
        """Genera copy publicitario optimizado para un canal especifico."""
        try:
            results = vector_store.similarity_search(product, k=3)
            product_info = "\n".join(sr.chunk.content[:300] for sr in results) if results else ""
            return (
                f"GENERA COPY PARA:\n"
                f"Producto: {product}\nInfo del producto: {product_info}\n"
                f"Audiencia: {audience}\nCanal: {channel}\nTono: {tone}\nObjetivo: {objective}\n\n"
                "INCLUYE:\n1. Headline (max 40 chars)\n2. Primary text (max 125 chars para feed)\n"
                "3. Description\n4. CTA especifico\n5. 5-8 hashtags relevantes\n"
                "6. Descripcion de imagen/video ideal"
            )
        except Exception as e:
            return f"Error generando copy: {e}"

    @tool
    def generate_ad_image(
        product_description: str,
        channel: str = "instagram_feed",
        brand_colors: str = "",
    ) -> str:
        """Genera una imagen publicitaria profesional con DALL-E 3."""
        if not dalle_adapter:
            return f"DALL-E no configurado. Sugerencia de imagen: {product_description}"
        colors = [c.strip() for c in brand_colors.split(",") if c.strip()] if brand_colors else None
        result = dalle_adapter.generate_ad_image(
            product_description=product_description, brand_colors=colors, channel=channel,
        )
        if result.get("error"):
            return f"Error generando imagen: {result['error']}"
        images = result.get("images", [])
        if not images:
            return "No se pudo generar la imagen."
        img = images[0]
        return (
            f"IMAGEN GENERADA:\n- Archivo: {img['filename']}\n- Path: {img['path']}\n"
            f"- Prompt usado: {img['revised_prompt']}\n- Tamano: {img['size']}\n"
            f"La imagen esta lista para usar en el ad."
        )

    @tool
    def generate_ab_variations(
        original_copy: str, variation_type: str = "headline", num_variations: int = 3,
    ) -> str:
        """Genera variaciones A/B de un copy para testear cual convierte mejor."""
        return (
            f"GENERA {num_variations} VARIACIONES A/B:\nTipo: {variation_type}\n"
            f"Original: {original_copy}\n\nREGLAS PARA VARIACIONES:\n"
            f"- Variacion A: Mas emocional/urgente\n- Variacion B: Mas racional/datos\n"
            f"- Variacion C: Mas casual/cercana\n"
            f"- Cada variacion debe ser significativamente diferente\n"
            f"- Mantener el mismo mensaje core pero diferente angulo\n"
            f"- Indicar para cada una por que podria funcionar mejor"
        )

    @tool
    def generate_video_script(
        product: str, platform: str = "instagram_reel", duration_seconds: int = 30,
    ) -> str:
        """Genera un script de video para Reels, TikTok o Stories."""
        try:
            results = vector_store.similarity_search(product, k=3)
            product_info = "\n".join(sr.chunk.content[:200] for sr in results) if results else ""
        except Exception:
            product_info = ""
        return (
            f"GENERA SCRIPT DE VIDEO:\nProducto: {product}\nInfo: {product_info}\n"
            f"Plataforma: {platform}\nDuracion: {duration_seconds} seg\n\n"
            f"ESTRUCTURA (formula viral):\n"
            f"1. HOOK (0-3s): Frase que detenga el scroll\n"
            f"2. PROBLEMA (3-8s): El pain point del cliente\n"
            f"3. SOLUCION (8-20s): Tu producto como la respuesta\n"
            f"4. PRUEBA: Beneficio concreto, testimonio o resultado\n"
            f"5. CTA: Accion clara\n\n"
            f"INCLUIR:\n- Texto en pantalla para cada escena\n"
            f"- Indicacion de musica/audio trending\n- Transiciones sugeridas"
        )

    @tool
    def generate_email_sequence(
        product: str, sequence_type: str = "sales", num_emails: int = 4,
    ) -> str:
        """Genera una secuencia de emails (funnel) para nurturing o ventas."""
        try:
            results = vector_store.similarity_search(product, k=3)
            product_info = "\n".join(sr.chunk.content[:200] for sr in results) if results else ""
        except Exception:
            product_info = ""

        templates = {
            "sales": [
                "Email 1 (Dia 0): Presentacion del problema + valor",
                "Email 2 (Dia 2): Caso de exito / testimonio",
                "Email 3 (Dia 4): Oferta especial + beneficios",
                "Email 4 (Dia 6): Urgencia/escasez + ultimo llamado",
            ],
            "welcome": [
                "Email 1 (Inmediato): Bienvenida + que esperar",
                "Email 2 (Dia 1): Tu historia / por que existimos",
                "Email 3 (Dia 3): Mejor producto / mas vendido",
                "Email 4 (Dia 5): Oferta de bienvenida exclusiva",
            ],
            "abandoned_cart": [
                "Email 1 (1 hora): Recordatorio amigable",
                "Email 2 (24 horas): Beneficios del producto",
                "Email 3 (48 horas): Descuento exclusivo",
                "Email 4 (72 horas): Ultimo aviso + urgencia",
            ],
            "launch": [
                "Email 1 (Dia -7): Teaser / expectativa",
                "Email 2 (Dia -3): Revelacion parcial + early access",
                "Email 3 (Dia 0): LANZAMIENTO + oferta especial",
                "Email 4 (Dia 2): Testimonios primeros compradores + FOMO",
            ],
        }
        structure = templates.get(sequence_type, templates["sales"])[:num_emails]
        return (
            f"GENERA SECUENCIA DE {num_emails} EMAILS ({sequence_type}):\n"
            f"Producto: {product}\nInfo: {product_info}\n\n"
            f"ESTRUCTURA:\n" + "\n".join(f"  {s}" for s in structure) + "\n\n"
            f"PARA CADA EMAIL INCLUIR:\n"
            f"- Subject line (max 50 chars, con emoji)\n- Preview text (max 90 chars)\n"
            f"- Body (hook, valor, CTA)\n- CTA principal (boton)\n- P.D. (gancho extra)\n\n"
            f"REGLAS:\n- Tono conversacional\n- Cada email debe funcionar solo\n"
            f"- Progresion: curiosidad → interes → deseo → accion"
        )

    return [generate_ad_copy, generate_ad_image, generate_ab_variations,
            generate_video_script, generate_email_sequence]
