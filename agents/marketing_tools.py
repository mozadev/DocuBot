"""Herramientas especializadas del agente de marketing LangGraph."""

from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import tool

from core.logger import logger


def build_marketing_tools(vector_store, tavily_adapter=None, dalle_adapter=None) -> list:
    """Construye herramientas de marketing inyectando vector store, Tavily y DALL-E."""

    @tool
    def search_product_catalog(query: str) -> str:
        """Busca productos, servicios, precios y descripciones en los documentos del negocio.
        Usa esta herramienta PRIMERO para entender que vende el negocio."""
        try:
            queries = [query, f"productos {query}", f"precios {query}", f"catalogo {query}"]
            all_results = []
            seen = set()
            for q in queries:
                results = vector_store.similarity_search(q, k=3)
                for sr in results:
                    key = f"{sr.chunk.filename}:{sr.chunk.content[:50]}"
                    if key not in seen:
                        seen.add(key)
                        all_results.append(sr)
            if not all_results:
                return f"No se encontraron productos o servicios relacionados con '{query}'."
            parts: List[str] = []
            for i, sr in enumerate(all_results[:8], 1):
                parts.append(
                    f"[Producto/Info {i} | {sr.chunk.filename} | score={sr.score:.3f}]\n"
                    f"{sr.chunk.content[:500]}"
                )
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.error(f"Error en search_product_catalog: {e}")
            return f"Error: {e}"

    @tool
    def analyze_business_data(data_description: str) -> str:
        """Analiza los datos de negocio proporcionados (ventas, WhatsApp, ads anteriores).
        Usa esta herramienta para extraer insights accionables de los datos del tenant.
        Pasa como input la descripcion de los datos que quieres analizar."""
        return (
            f"Datos para analizar: {data_description}\n\n"
            "INSTRUCCIONES: Con estos datos, identifica:\n"
            "1. Productos estrella (top sellers)\n"
            "2. Horarios de mayor actividad\n"
            "3. Preguntas frecuentes que indican necesidades no cubiertas\n"
            "4. Ticket promedio para definir presupuesto de ads\n"
            "5. Tasa de conversion actual para establecer KPIs realistas\n"
            "6. Oportunidades de mejora vs ads anteriores"
        )

    @tool
    def research_audience(business_type: str, location: str = "", current_audience: str = "") -> str:
        """Investiga y define el publico objetivo ideal basandose en el tipo de negocio.
        Usa esta herramienta para definir targeting preciso de Facebook/Instagram Ads."""
        try:
            results = vector_store.similarity_search(
                f"clientes publico objetivo {business_type}", k=4
            )
            doc_context = ""
            if results:
                doc_context = "\n".join(
                    f"- [{sr.chunk.filename}]: {sr.chunk.content[:300]}"
                    for sr in results
                )

            return (
                f"INVESTIGACION DE AUDIENCIA:\n"
                f"Tipo de negocio: {business_type}\n"
                f"Ubicacion: {location or 'No especificada'}\n"
                f"Audiencia actual conocida: {current_audience or 'No especificada'}\n"
                f"Informacion de documentos:\n{doc_context}\n\n"
                "DEFINE:\n"
                "1. Segmento primario (edad, genero, ubicacion, intereses)\n"
                "2. Segmento secundario (lookalike)\n"
                "3. Segmento de retargeting\n"
                "4. Intereses especificos para Meta Ads targeting\n"
                "5. Comportamientos de compra relevantes"
            )
        except Exception as e:
            return f"Error investigando audiencia: {e}"

    @tool
    def generate_ad_copy(
        product: str,
        audience: str,
        channel: str,
        tone: str = "profesional",
        objective: str = "conversion",
    ) -> str:
        """Genera copy publicitario optimizado para un canal especifico.
        Usa esta herramienta para crear el texto de cada pieza de la campana."""
        try:
            results = vector_store.similarity_search(product, k=3)
            product_info = ""
            if results:
                product_info = "\n".join(sr.chunk.content[:300] for sr in results)

            return (
                f"GENERA COPY PARA:\n"
                f"Producto: {product}\n"
                f"Info del producto: {product_info}\n"
                f"Audiencia: {audience}\n"
                f"Canal: {channel}\n"
                f"Tono: {tone}\n"
                f"Objetivo: {objective}\n\n"
                "INCLUYE:\n"
                "1. Headline (max 40 chars)\n"
                "2. Primary text (max 125 chars para feed, max 90 para stories)\n"
                "3. Description\n"
                "4. CTA especifico\n"
                "5. 5-8 hashtags relevantes\n"
                "6. Descripcion de imagen/video ideal"
            )
        except Exception as e:
            return f"Error generando copy: {e}"

    @tool
    def calculate_budget(
        objective: str,
        avg_cpc: float = 0.0,
        avg_cpm: float = 0.0,
        target_conversions: int = 0,
        duration_days: int = 7,
        currency: str = "USD",
    ) -> str:
        """Calcula el presupuesto recomendado para una campana de Meta Ads.
        Si tienes CPC/CPM reales del tenant, pasa esos datos para un calculo preciso."""
        if avg_cpc > 0 and target_conversions > 0:
            estimated_budget = avg_cpc * target_conversions
            daily = estimated_budget / duration_days if duration_days > 0 else estimated_budget
            return (
                f"CALCULO BASADO EN DATOS REALES:\n"
                f"CPC promedio: ${avg_cpc:.2f} {currency}\n"
                f"Conversiones objetivo: {target_conversions}\n"
                f"Presupuesto total estimado: ${estimated_budget:,.2f} {currency}\n"
                f"Presupuesto diario: ${daily:,.2f} {currency}\n"
                f"Duracion: {duration_days} dias\n"
                f"Nota: Basado en rendimiento historico real del tenant."
            )
        elif avg_cpm > 0:
            impressions = 50000
            estimated_budget = (avg_cpm / 1000) * impressions
            daily = estimated_budget / duration_days if duration_days > 0 else estimated_budget
            return (
                f"CALCULO BASADO EN CPM REAL:\n"
                f"CPM promedio: ${avg_cpm:.2f} {currency}\n"
                f"Impresiones estimadas: {impressions:,}\n"
                f"Presupuesto total: ${estimated_budget:,.2f} {currency}\n"
                f"Presupuesto diario: ${daily:,.2f} {currency}\n"
                f"Duracion: {duration_days} dias"
            )
        else:
            return (
                f"ESTIMACION GENERICA (sin datos historicos):\n"
                f"Objetivo: {objective}\n"
                f"Duracion: {duration_days} dias\n"
                f"Rango recomendado LATAM:\n"
                f"  - Awareness: $3-5 {currency}/dia\n"
                f"  - Traffic: $5-10 {currency}/dia\n"
                f"  - Conversiones: $10-25 {currency}/dia\n"
                f"  - Lead generation: $8-15 {currency}/dia\n"
                f"IMPORTANTE: Estos son estimados genericos. "
                f"Se recomienda conectar Meta Ads para datos reales de CPC/CPM."
            )

    @tool
    def plan_content_calendar(
        campaign_duration_days: int,
        channels: str,
        num_pieces: int = 10,
    ) -> str:
        """Planifica un calendario de contenido para la campana.
        Usa esta herramienta para definir cuando publicar cada pieza."""
        return (
            f"PLANIFICA CALENDARIO:\n"
            f"Duracion: {campaign_duration_days} dias\n"
            f"Canales: {channels}\n"
            f"Piezas a distribuir: {num_pieces}\n\n"
            "REGLAS DE CALENDARIO PARA LATAM:\n"
            "- Facebook Feed: mejor entre 12-14h y 19-21h\n"
            "- Instagram Feed: mejor entre 11-13h y 18-20h\n"
            "- Instagram Stories: mejor entre 8-10h y 20-22h\n"
            "- Email: mejor martes a jueves 9-11h\n"
            "- WhatsApp broadcast: respetar horario comercial 9-18h\n"
            "- Frecuencia Facebook/IG: 1 post/dia max, 3-5 stories/dia\n"
            "- No publicar los mismos dias en todos los canales (distribuir)\n"
            "- Si tienes horas pico de WhatsApp del tenant, priorizar esos horarios"
        )

    @tool
    def review_campaign_quality(campaign_summary: str) -> str:
        """Revisa la calidad de la campana generada y sugiere mejoras.
        Usa esta herramienta como ultimo paso para asegurar calidad."""
        return (
            f"REVISA ESTA CAMPANA:\n{campaign_summary}\n\n"
            "CHECKLIST DE CALIDAD:\n"
            "1. Cada pieza tiene headline < 40 chars?\n"
            "2. Los CTAs son claros y accionables?\n"
            "3. Los hashtags son relevantes (no genericos)?\n"
            "4. El tono es consistente en todas las piezas?\n"
            "5. Hay variedad de formatos (feed, stories, carousel, reel)?\n"
            "6. El presupuesto es realista para el mercado LATAM?\n"
            "7. El calendario tiene buena distribucion?\n"
            "8. Los KPIs son medibles y alcanzables?\n"
            "9. Las imagenes sugeridas son coherentes con la marca?\n"
            "10. Hay al menos un CTA hacia WhatsApp?\n"
            "Si algo falla, corrige y mejora."
        )

    # ---- Herramientas de busqueda web (Tavily) ----

    @tool
    def search_market_trends(industry: str, location: str = "Latinoamerica") -> str:
        """Busca tendencias REALES de marketing digital en internet para una industria.
        Usa esta herramienta para obtener datos actualizados del mercado.
        Requiere Tavily API key configurada."""
        if not tavily_adapter or not tavily_adapter.is_enabled:
            return (
                "Busqueda web no disponible (Tavily no configurado). "
                "Usa conocimiento general para estimar tendencias."
            )
        result = tavily_adapter.search_market_trends(industry, location)
        if result.get("error"):
            return f"Error en busqueda: {result['error']}"
        parts = []
        if result.get("answer"):
            parts.append(f"RESUMEN: {result['answer']}")
        for r in result.get("results", []):
            parts.append(f"[{r['title']}] ({r['url']})\n{r['content'][:300]}")
        return "\n\n---\n\n".join(parts) if parts else "Sin resultados."

    @tool
    def search_competitors(business_description: str, location: str = "") -> str:
        """Busca informacion REAL de competidores en internet.
        Usa esta herramienta para analizar la competencia y diferenciarse.
        Requiere Tavily API key configurada."""
        if not tavily_adapter or not tavily_adapter.is_enabled:
            return (
                "Busqueda web no disponible (Tavily no configurado). "
                "Diferencia el negocio usando la informacion disponible en los documentos."
            )
        result = tavily_adapter.search_competitors(business_description, location)
        if result.get("error"):
            return f"Error en busqueda: {result['error']}"
        parts = []
        if result.get("answer"):
            parts.append(f"RESUMEN DE COMPETENCIA: {result['answer']}")
        for r in result.get("results", []):
            parts.append(f"[{r['title']}] ({r['url']})\n{r['content'][:300]}")
        return "\n\n---\n\n".join(parts) if parts else "Sin resultados."

    @tool
    def search_ad_benchmarks(industry: str, platform: str = "Facebook Ads") -> str:
        """Busca benchmarks REALES de CPC, CTR, CPM para una industria en internet.
        Usa esta herramienta para fundamentar recomendaciones de presupuesto con datos reales.
        Requiere Tavily API key configurada."""
        if not tavily_adapter or not tavily_adapter.is_enabled:
            return (
                "Busqueda web no disponible (Tavily no configurado). "
                "Usa estos benchmarks genericos para LATAM:\n"
                "- CPC promedio Facebook: $0.20-0.80 USD\n"
                "- CTR promedio Facebook: 1.5-3.0%\n"
                "- CPM promedio Facebook: $3-8 USD\n"
                "- CPC promedio Instagram: $0.30-1.00 USD\n"
                "NOTA: Estos son estimados genericos, no datos reales de la industria."
            )
        result = tavily_adapter.search_ad_benchmarks(industry, platform)
        if result.get("error"):
            return f"Error en busqueda: {result['error']}"
        parts = []
        if result.get("answer"):
            parts.append(f"BENCHMARKS REALES: {result['answer']}")
        for r in result.get("results", []):
            parts.append(f"[{r['title']}] ({r['url']})\n{r['content'][:300]}")
        return "\n\n---\n\n".join(parts) if parts else "Sin resultados de benchmarks."

    @tool
    def search_content_ideas(topic: str, channel: str = "Instagram") -> str:
        """Busca ideas de contenido VIRAL y TRENDING en internet para un tema.
        Usa esta herramienta para crear contenido que siga las tendencias actuales.
        Requiere Tavily API key configurada."""
        if not tavily_adapter or not tavily_adapter.is_enabled:
            return (
                "Busqueda web no disponible (Tavily no configurado). "
                "Genera ideas de contenido con tu conocimiento de marketing."
            )
        result = tavily_adapter.search_content_ideas(topic, channel)
        if result.get("error"):
            return f"Error en busqueda: {result['error']}"
        parts = []
        if result.get("answer"):
            parts.append(f"IDEAS TRENDING: {result['answer']}")
        for r in result.get("results", []):
            parts.append(f"[{r['title']}]\n{r['content'][:300]}")
        return "\n\n---\n\n".join(parts) if parts else "Sin ideas encontradas."

    # ---- Generacion de imagenes (DALL-E) ----

    @tool
    def generate_ad_image(
        product_description: str,
        channel: str = "instagram_feed",
        brand_colors: str = "",
    ) -> str:
        """Genera una imagen publicitaria profesional con DALL-E 3.
        Usa esta herramienta cuando necesites crear el visual de un ad.
        Channels: instagram_feed, instagram_story, facebook_feed, ad_landscape, ad_square.
        brand_colors: colores separados por coma, ej: '#FF6B9D, #2D1B69'."""
        if not dalle_adapter:
            return (
                f"Generacion de imagenes no disponible (DALL-E no configurado). "
                f"Sugerencia de imagen: {product_description}"
            )
        colors = [c.strip() for c in brand_colors.split(",") if c.strip()] if brand_colors else None
        result = dalle_adapter.generate_ad_image(
            product_description=product_description,
            brand_colors=colors,
            channel=channel,
        )
        if result.get("error"):
            return f"Error generando imagen: {result['error']}"
        images = result.get("images", [])
        if not images:
            return "No se pudo generar la imagen."
        img = images[0]
        return (
            f"IMAGEN GENERADA:\n"
            f"- Archivo: {img['filename']}\n"
            f"- Path: {img['path']}\n"
            f"- Prompt usado: {img['revised_prompt']}\n"
            f"- Tamano: {img['size']}\n"
            f"La imagen esta lista para usar en el ad."
        )

    # ---- Variaciones A/B ----

    @tool
    def generate_ab_variations(
        original_copy: str,
        variation_type: str = "headline",
        num_variations: int = 3,
    ) -> str:
        """Genera variaciones A/B de un copy para testear cual convierte mejor.
        variation_type: headline | body | cta | full
        Usa esta herramienta despues de crear el contenido principal para ofrecer alternativas."""
        return (
            f"GENERA {num_variations} VARIACIONES A/B:\n"
            f"Tipo: {variation_type}\n"
            f"Original: {original_copy}\n\n"
            f"REGLAS PARA VARIACIONES:\n"
            f"- Variacion A: Mas emocional/urgente\n"
            f"- Variacion B: Mas racional/datos\n"
            f"- Variacion C: Mas casual/cercana\n"
            f"- Cada variacion debe ser significativamente diferente\n"
            f"- Mantener el mismo mensaje core pero diferente angulo\n"
            f"- Indicar para cada una por que podria funcionar mejor"
        )

    # ---- Video Scripts ----

    @tool
    def generate_video_script(
        product: str,
        platform: str = "instagram_reel",
        duration_seconds: int = 30,
    ) -> str:
        """Genera un script de video para Reels, TikTok o Stories.
        Usa esta herramienta cuando la campana incluya contenido en video.
        platform: instagram_reel | tiktok | instagram_story | facebook_reel."""
        try:
            results = vector_store.similarity_search(product, k=3)
            product_info = "\n".join(sr.chunk.content[:200] for sr in results) if results else ""
        except Exception:
            product_info = ""

        return (
            f"GENERA SCRIPT DE VIDEO:\n"
            f"Producto: {product}\n"
            f"Info del producto: {product_info}\n"
            f"Plataforma: {platform}\n"
            f"Duracion: {duration_seconds} segundos\n\n"
            f"ESTRUCTURA OBLIGATORIA (formula viral):\n"
            f"1. HOOK (0-3s): Frase que detenga el scroll. Pregunta o dato impactante\n"
            f"2. PROBLEMA (3-8s): El pain point del cliente\n"
            f"3. SOLUCION (8-20s): Tu producto como la respuesta\n"
            f"4. PRUEBA ({20 if duration_seconds > 20 else duration_seconds-5}-{duration_seconds-5}s): "
            f"Beneficio concreto, testimonio o resultado\n"
            f"5. CTA ({duration_seconds-5}-{duration_seconds}s): Accion clara\n\n"
            f"INCLUIR:\n"
            f"- Texto en pantalla (caption overlay) para cada escena\n"
            f"- Indicacion de musica/audio trending\n"
            f"- Transiciones sugeridas\n"
            f"- Indicar si necesita cara de persona o solo producto"
        )

    # ---- Email Sequences ----

    @tool
    def generate_email_sequence(
        product: str,
        sequence_type: str = "sales",
        num_emails: int = 4,
    ) -> str:
        """Genera una secuencia de emails (funnel) para nurturing o ventas.
        sequence_type: sales | welcome | abandoned_cart | reengagement | launch.
        Usa esta herramienta cuando la campana incluya email marketing."""
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
            f"Producto: {product}\n"
            f"Info: {product_info}\n\n"
            f"ESTRUCTURA:\n" + "\n".join(f"  {s}" for s in structure) + "\n\n"
            f"PARA CADA EMAIL INCLUIR:\n"
            f"- Subject line (max 50 chars, con emoji)\n"
            f"- Preview text (max 90 chars)\n"
            f"- Body (con estructura: hook, valor, CTA)\n"
            f"- CTA principal (boton)\n"
            f"- P.D. (postdata con gancho extra)\n\n"
            f"REGLAS:\n"
            f"- Tono conversacional, como hablar con un amigo\n"
            f"- Cada email debe funcionar solo (si no leyeron los anteriores)\n"
            f"- Progresion de urgencia: curiosidad → interes → deseo → accion\n"
            f"- Subject lines que generen apertura (pregunta, numero, curiosidad)"
        )

    # ---- Multi-idioma ----

    @tool
    def translate_campaign_content(
        content: str,
        target_language: str = "en",
        adapt_culturally: bool = True,
    ) -> str:
        """Traduce y adapta culturalmente contenido de marketing a otro idioma.
        Usa esta herramienta cuando necesites versiones en multiples idiomas.
        target_language: en (ingles), pt (portugues), fr (frances), etc.
        Si adapt_culturally=True, no traduce literal sino que adapta modismos y CTAs."""
        return (
            f"TRADUCE Y ADAPTA:\n"
            f"Idioma destino: {target_language}\n"
            f"Adaptacion cultural: {'Si' if adapt_culturally else 'No (literal)'}\n"
            f"Contenido original:\n{content}\n\n"
            f"REGLAS:\n"
            f"- NO traducir literalmente, ADAPTAR al mercado del idioma destino\n"
            f"- Adaptar modismos, referencias culturales y humor\n"
            f"- Mantener el tono y la intencion del mensaje\n"
            f"- Adaptar hashtags al idioma destino\n"
            f"- Si es para Brasil (pt-BR): adaptar a cultura brasilena\n"
            f"- Si es para USA (en): adaptar a cultura americana\n"
            f"- Mantener emojis y formato"
        )

    # ---- Analisis de rendimiento (performance learning) ----

    @tool
    def analyze_past_performance(performance_data: str) -> str:
        """Analiza el rendimiento de campanas/ads anteriores para aprender que funciono.
        Usa esta herramienta cuando tengas datos de rendimiento de Meta Ads anteriores.
        Pasa los datos como texto con metricas (CTR, CPC, conversiones, etc.)."""
        return (
            f"ANALIZA RENDIMIENTO PASADO:\n{performance_data}\n\n"
            f"EXTRAE:\n"
            f"1. Que tipo de contenido tuvo mejor CTR? (copy, formato, canal)\n"
            f"2. Que audiencia convirtio mejor?\n"
            f"3. Que horarios funcionaron?\n"
            f"4. Que CPC/CPM es realista para este negocio?\n"
            f"5. Que NO funciono y debemos evitar?\n"
            f"6. Recomendaciones concretas para la proxima campana\n\n"
            f"APLICA ESTAS LECCIONES en todo el contenido que generes."
        )

    @tool
    def suggest_optimization(current_metrics: str, target_kpi: str = "CTR") -> str:
        """Sugiere optimizaciones para mejorar un KPI especifico basandose en metricas actuales.
        Usa cuando NestJS reporte que una campana activa no alcanza sus metas."""
        return (
            f"OPTIMIZA PARA MEJORAR {target_kpi}:\n"
            f"Metricas actuales:\n{current_metrics}\n\n"
            f"SUGIERE CAMBIOS EN:\n"
            f"1. Copy: nuevo headline o body que pueda mejorar {target_kpi}\n"
            f"2. Audiencia: ajustar targeting\n"
            f"3. Presupuesto: redistribuir entre ad sets\n"
            f"4. Creative: nueva imagen o formato\n"
            f"5. Schedule: cambiar horarios de publicacion\n"
            f"6. CTA: probar otro call to action\n"
            f"Prioriza los cambios de mayor impacto primero."
        )

    # ---- Multi-plataforma ----

    @tool
    def adapt_for_google_ads(content: str, campaign_type: str = "search") -> str:
        """Adapta contenido de marketing para Google Ads (Search, Display, Shopping).
        campaign_type: search | display | shopping | youtube.
        Usa cuando la campana incluya Google Ads ademas de Meta."""
        limits = {
            "search": "Headlines: max 30 chars (x15). Descriptions: max 90 chars (x4).",
            "display": "Headline: max 30 chars. Long headline: max 90 chars. Description: max 90 chars.",
            "shopping": "Title: max 150 chars. Description: max 5000 chars.",
            "youtube": "Headline: max 15 chars. Long headline: max 90 chars. Description: max 70 chars.",
        }
        return (
            f"ADAPTA PARA GOOGLE ADS ({campaign_type.upper()}):\n"
            f"Contenido original (Meta):\n{content}\n\n"
            f"LIMITES DE GOOGLE ADS {campaign_type.upper()}:\n"
            f"{limits.get(campaign_type, limits['search'])}\n\n"
            f"REGLAS GOOGLE ADS:\n"
            f"- Google Ads es mas directo/informativo que Meta (el usuario ya esta buscando)\n"
            f"- Incluir keywords del negocio en los headlines\n"
            f"- Search: foco en intencion de compra, no engagement\n"
            f"- Display: visual + brand awareness\n"
            f"- Incluir extensiones: sitelinks, callouts, precio\n"
            f"- Generar multiples variaciones de headlines para que Google optimice"
        )

    @tool
    def adapt_for_tiktok(content: str, trend_style: str = "storytelling") -> str:
        """Adapta contenido de marketing para TikTok Ads.
        trend_style: storytelling | ugc | educational | meme | challenge.
        TikTok requiere un tono completamente diferente a Meta."""
        return (
            f"ADAPTA PARA TIKTOK ADS:\n"
            f"Contenido original (Meta):\n{content}\n\n"
            f"Estilo trending: {trend_style}\n\n"
            f"REGLAS TIKTOK:\n"
            f"- NO parece publicidad. Debe parecer contenido organico de un creator\n"
            f"- Hook en los primeros 2 segundos o pierdes al usuario\n"
            f"- Formato vertical 9:16\n"
            f"- Duracion ideal: 15-30 segundos\n"
            f"- Usa trending sounds/music (indica cual)\n"
            f"- Texto en pantalla es OBLIGATORIO (muchos ven sin sonido)\n"
            f"- Storytelling: problema → solucion → resultado\n"
            f"- UGC style: que parezca grabado con celular, no produccion\n"
            f"- CTA: 'Link en bio' o 'Comentame X para info'\n"
            f"- NO hashtags corporativos, usa trending hashtags de TikTok"
        )

    # ── SEO Content ──

    @tool
    def research_seo_keywords(
        business_type: str,
        location: str = "",
        language: str = "es",
    ) -> str:
        """Investiga keywords SEO para el negocio.
        Retorna keywords primarias, secundarias, long-tail, y volumen estimado.
        Usa esto ANTES de generar contenido SEO."""
        web_results = ""
        if tavily_adapter and tavily_adapter._api_key:
            try:
                web_results = tavily_adapter.search(
                    f"mejores keywords SEO para {business_type} {location} {language} 2026"
                )
            except Exception:
                web_results = ""

        return (
            f"INVESTIGA KEYWORDS SEO:\n"
            f"Negocio: {business_type}\n"
            f"Ubicacion: {location}\n"
            f"Idioma: {language}\n"
            f"Datos web: {web_results[:500] if web_results else 'No disponible'}\n\n"
            f"GENERA:\n"
            f"1. 5 keywords primarias (alto volumen, alta competencia)\n"
            f"2. 10 keywords secundarias (medio volumen)\n"
            f"3. 15 keywords long-tail (bajo volumen, baja competencia, alta conversion)\n"
            f"4. Intento de busqueda de cada keyword (informacional/transaccional/navegacional)\n"
            f"5. Dificultad estimada (facil/media/dificil)\n"
            f"Formato: tabla con keyword | volumen_estimado | dificultad | intento"
        )

    @tool
    def generate_seo_blog_post(
        topic: str,
        primary_keyword: str,
        secondary_keywords: str = "",
        word_count: int = 1500,
        tone: str = "profesional",
    ) -> str:
        """Genera un blog post completo optimizado para SEO.
        Incluye titulo H1, meta description, estructura H2/H3, internal linking suggestions."""
        return (
            f"GENERA BLOG POST SEO-OPTIMIZADO:\n"
            f"Tema: {topic}\n"
            f"Keyword primaria: {primary_keyword}\n"
            f"Keywords secundarias: {secondary_keywords}\n"
            f"Palabras: ~{word_count}\n"
            f"Tono: {tone}\n\n"
            f"ESTRUCTURA OBLIGATORIA:\n"
            f"- Titulo H1: incluir keyword primaria, <60 chars, atractivo para clicks\n"
            f"- Meta description: 150-160 chars, incluir keyword, con CTA\n"
            f"- Intro: hook + keyword en primer parrafo + preview del contenido\n"
            f"- 4-6 secciones H2 (incluir keywords secundarias naturalmente)\n"
            f"- Sub-secciones H3 donde tenga sentido\n"
            f"- Listas con bullets (Google las prefiere para featured snippets)\n"
            f"- FAQ section al final (3-5 preguntas = oportunidad de featured snippet)\n"
            f"- Conclusion con CTA\n"
            f"- Sugerencias de internal links: [anchortext](URL_sugerida)\n\n"
            f"REGLAS SEO:\n"
            f"- Densidad keyword primaria: 1-2% (natural, no spam)\n"
            f"- Usar keyword en primer H2 y ultimo H2\n"
            f"- Alt text sugerido para imagenes\n"
            f"- Schema markup suggestion (Article, FAQ, HowTo)\n"
            f"- Readability: oraciones cortas, parrafos de 2-3 lineas max"
        )

    @tool
    def generate_meta_tags(
        page_title: str,
        page_description: str,
        primary_keyword: str,
    ) -> str:
        """Genera meta tags SEO optimizados para una pagina web.
        Title tag, meta description, Open Graph, Twitter Cards."""
        return (
            f"GENERA META TAGS SEO:\n"
            f"Pagina: {page_title}\n"
            f"Descripcion: {page_description}\n"
            f"Keyword: {primary_keyword}\n\n"
            f"GENERA:\n"
            f"1. <title> tag: keyword al inicio, marca al final, <60 chars\n"
            f"2. <meta description>: 150-160 chars, keyword natural, CTA, unique\n"
            f"3. <meta keywords>: 5-8 keywords relevantes\n"
            f"4. Open Graph tags (og:title, og:description, og:type, og:image suggestion)\n"
            f"5. Twitter Card tags\n"
            f"6. Schema.org JSON-LD suggestion para el tipo de pagina\n"
            f"Formato: HTML listo para copiar/pegar"
        )

    @tool
    def check_content_seo_score(content: str, target_keyword: str) -> str:
        """Analiza contenido existente y calcula un SEO score con recomendaciones.
        Retorna score 0-100 y accionables para mejorar."""
        word_count = len(content.split())
        keyword_count = content.lower().count(target_keyword.lower())
        density = (keyword_count / max(word_count, 1)) * 100

        has_h2 = "##" in content or "<h2" in content.lower()
        has_lists = "- " in content or "* " in content or "<li" in content.lower()
        has_links = "[" in content and "](" in content or "<a " in content.lower()

        return (
            f"ANALISIS SEO DEL CONTENIDO:\n"
            f"Palabras: {word_count}\n"
            f"Keyword '{target_keyword}' aparece: {keyword_count} veces\n"
            f"Densidad keyword: {density:.1f}%\n"
            f"Tiene H2s: {'Si' if has_h2 else 'No'}\n"
            f"Tiene listas: {'Si' if has_lists else 'No'}\n"
            f"Tiene links: {'Si' if has_links else 'No'}\n\n"
            f"EVALUA Y CALIFICA 0-100:\n"
            f"- Titulo (keyword, longitud, atractivo): /20\n"
            f"- Densidad keyword (ideal 1-2%): /15\n"
            f"- Estructura (H2, H3, listas): /20\n"
            f"- Longitud (ideal >1000 palabras): /15\n"
            f"- Readability (oraciones cortas, simple): /15\n"
            f"- Links internos/externos: /15\n"
            f"Total: /100\n\n"
            f"RECOMENDACIONES (maximo 5 accionables prioritarios)"
        )

    # ── Plagiarism Check ──

    @tool
    def check_plagiarism(content: str, content_type: str = "ad_copy") -> str:
        """Verifica originalidad del contenido.
        Busca frases clave en web para detectar si es copia de otro ad/blog.
        content_type: ad_copy | blog_post | email | social_post"""
        sentences = [s.strip() for s in content.replace("\n", ". ").split(". ") if len(s.strip()) > 20]
        check_sentences = sentences[:5]

        web_matches = []
        if tavily_adapter and tavily_adapter._api_key:
            for sentence in check_sentences[:3]:
                try:
                    results = tavily_adapter.search(f'"{sentence}"')
                    if results and "no results" not in results.lower():
                        web_matches.append({"sentence": sentence, "web_result": results[:300]})
                except Exception:
                    pass

        return (
            f"ANALISIS DE ORIGINALIDAD:\n"
            f"Tipo: {content_type}\n"
            f"Oraciones analizadas: {len(check_sentences)}\n"
            f"Coincidencias web encontradas: {len(web_matches)}\n\n"
            f"{'COINCIDENCIAS:' if web_matches else 'No se encontraron copias directas.'}\n"
            + "\n".join(f"- '{m['sentence']}' → {m['web_result']}" for m in web_matches)
            + f"\n\nEVALUA:\n"
            f"- Originalidad estimada: X/100\n"
            f"- Riesgo de plagio: bajo/medio/alto\n"
            f"- Si hay coincidencias, sugiere reescribir las frases similares\n"
            f"- Verifica que CTAs y slogans no sean de competidores conocidos"
        )

    # ── LinkedIn Ads (B2B) ──

    @tool
    def adapt_for_linkedin(
        content: str,
        objective: str = "lead_generation",
        ad_format: str = "single_image",
    ) -> str:
        """Adapta contenido de marketing para LinkedIn Ads (B2B).
        objective: lead_generation | brand_awareness | website_visits | engagement.
        ad_format: single_image | carousel | video | text_ad | message_ad | document_ad."""
        return (
            f"ADAPTA PARA LINKEDIN ADS (B2B):\n"
            f"Contenido original:\n{content}\n\n"
            f"Objetivo: {objective}\n"
            f"Formato: {ad_format}\n\n"
            f"REGLAS LINKEDIN:\n"
            f"- Tono PROFESIONAL. Nada de emojis excesivos ni lenguaje casual\n"
            f"- Enfocate en ROI, eficiencia, resultados medibles, casos de exito\n"
            f"- Headline: maximo 70 chars, directo al pain point del decision-maker\n"
            f"- Intro text: maximo 150 chars para el preview (antes del 'ver mas')\n"
            f"- Body: datos, estadisticas, social proof ('500+ empresas confian en...')\n"
            f"- CTA profesional: 'Solicitar demo' / 'Descargar whitepaper' / 'Agendar reunion'\n"
            f"- NO 'Compra ya' ni urgencia artificial. LinkedIn penaliza eso\n\n"
            f"TARGETING SUGERIDO:\n"
            f"- Job titles relevantes (CEO, Director, Gerente de...)\n"
            f"- Industrias objetivo\n"
            f"- Tamano de empresa (1-50, 51-200, 201-1000, 1000+)\n"
            f"- Seniority level\n\n"
            f"FORMATOS ESPECIFICOS:\n"
            f"- single_image: imagen profesional, no stock generico. Texto en imagen minimo\n"
            f"- carousel: 3-5 slides tipo presentacion ejecutiva\n"
            f"- video: 30-90 seg, subtitulos obligatorios, testimonial o demo\n"
            f"- text_ad: headline 25 chars + descripcion 75 chars (muy corto)\n"
            f"- message_ad: InMail personalizado, como si fuera de persona a persona\n"
            f"- document_ad: PDF descargable (lead magnet), titulo atractivo\n\n"
            f"METRICAS LINKEDIN:\n"
            f"- CTR promedio: 0.4-0.6% (mas bajo que Meta pero leads mas calificados)\n"
            f"- CPC promedio: $5-12 USD (mas caro pero mayor valor por lead)\n"
            f"- Mejor dia: martes a jueves, 8-10am horario del target"
        )

    # ── Retornar todas las herramientas ──

    tools = [
        # Investigacion
        search_product_catalog,
        analyze_business_data,
        research_audience,
        # Estrategia
        calculate_budget,
        plan_content_calendar,
        # Creacion
        generate_ad_copy,
        generate_ad_image,
        generate_ab_variations,
        generate_video_script,
        generate_email_sequence,
        # Multi-idioma
        translate_campaign_content,
        # Performance learning
        analyze_past_performance,
        suggest_optimization,
        # Multi-plataforma
        adapt_for_google_ads,
        adapt_for_tiktok,
        adapt_for_linkedin,
        # SEO
        research_seo_keywords,
        generate_seo_blog_post,
        generate_meta_tags,
        check_content_seo_score,
        # Plagiarism
        check_plagiarism,
        # Calidad
        review_campaign_quality,
        # Web (Tavily)
        search_market_trends,
        search_competitors,
        search_ad_benchmarks,
        search_content_ideas,
    ]
    return tools
