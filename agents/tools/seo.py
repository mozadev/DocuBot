"""Tools de SEO: keyword research, blog generation, meta tags, scoring."""

from __future__ import annotations
from langchain_core.tools import tool


def build_seo_tools(tavily_adapter=None) -> list:

    @tool
    def research_seo_keywords(business_type: str, location: str = "", language: str = "es") -> str:
        """Investiga keywords SEO: primarias, secundarias, long-tail."""
        web_results = ""
        if tavily_adapter and tavily_adapter._api_key:
            try:
                web_results = tavily_adapter.search(
                    f"mejores keywords SEO para {business_type} {location} {language} 2026"
                )
            except Exception:
                web_results = ""
        return (
            f"INVESTIGA KEYWORDS SEO:\nNegocio: {business_type}\n"
            f"Ubicacion: {location}\nIdioma: {language}\n"
            f"Datos web: {web_results[:500] if web_results else 'No disponible'}\n\n"
            f"GENERA:\n1. 5 keywords primarias (alto volumen)\n"
            f"2. 10 keywords secundarias\n3. 15 keywords long-tail\n"
            f"4. Intento de busqueda de cada keyword\n5. Dificultad estimada\n"
            f"Formato: tabla con keyword | volumen | dificultad | intento"
        )

    @tool
    def generate_seo_blog_post(
        topic: str, primary_keyword: str, secondary_keywords: str = "",
        word_count: int = 1500, tone: str = "profesional",
    ) -> str:
        """Genera blog post completo optimizado para SEO con H1, meta description, FAQ."""
        return (
            f"GENERA BLOG POST SEO-OPTIMIZADO:\nTema: {topic}\n"
            f"Keyword primaria: {primary_keyword}\nSecundarias: {secondary_keywords}\n"
            f"Palabras: ~{word_count}\nTono: {tone}\n\n"
            f"ESTRUCTURA:\n- H1: keyword, <60 chars\n- Meta description: 150-160 chars\n"
            f"- Intro: hook + keyword\n- 4-6 H2s\n- FAQ section (3-5 preguntas)\n"
            f"- Conclusion con CTA\n\nREGLAS SEO:\n- Densidad keyword: 1-2%\n"
            f"- Alt text para imagenes\n- Schema markup suggestion\n"
            f"- Readability: oraciones cortas, parrafos de 2-3 lineas"
        )

    @tool
    def generate_meta_tags(page_title: str, page_description: str, primary_keyword: str) -> str:
        """Genera meta tags SEO: title, description, OG, Twitter Cards, Schema.org."""
        return (
            f"GENERA META TAGS SEO:\nPagina: {page_title}\n"
            f"Descripcion: {page_description}\nKeyword: {primary_keyword}\n\n"
            f"GENERA:\n1. <title> tag (<60 chars)\n2. <meta description> (150-160 chars)\n"
            f"3. Open Graph tags\n4. Twitter Card tags\n5. Schema.org JSON-LD\n"
            f"Formato: HTML listo para copiar"
        )

    @tool
    def check_content_seo_score(content: str, target_keyword: str) -> str:
        """Analiza contenido y calcula SEO score 0-100 con recomendaciones."""
        word_count = len(content.split())
        keyword_count = content.lower().count(target_keyword.lower())
        density = (keyword_count / max(word_count, 1)) * 100
        has_h2 = "##" in content or "<h2" in content.lower()
        has_lists = "- " in content or "* " in content or "<li" in content.lower()
        has_links = "[" in content and "](" in content or "<a " in content.lower()
        return (
            f"ANALISIS SEO:\nPalabras: {word_count}\n"
            f"Keyword '{target_keyword}': {keyword_count} veces ({density:.1f}%)\n"
            f"H2s: {'Si' if has_h2 else 'No'} | Listas: {'Si' if has_lists else 'No'} | "
            f"Links: {'Si' if has_links else 'No'}\n\n"
            f"CALIFICA 0-100:\n- Titulo /20\n- Densidad keyword /15\n"
            f"- Estructura /20\n- Longitud /15\n- Readability /15\n- Links /15\n"
            f"RECOMENDACIONES (5 accionables)"
        )

    return [research_seo_keywords, generate_seo_blog_post, generate_meta_tags, check_content_seo_score]
