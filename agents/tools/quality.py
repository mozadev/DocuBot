"""Tools de calidad: plagiarism check."""

from __future__ import annotations
from langchain_core.tools import tool


def build_quality_tools(tavily_adapter=None) -> list:

    @tool
    def check_plagiarism(content: str, content_type: str = "ad_copy") -> str:
        """Verifica originalidad del contenido buscando frases en web."""
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
            f"ANALISIS DE ORIGINALIDAD:\nTipo: {content_type}\n"
            f"Oraciones analizadas: {len(check_sentences)}\n"
            f"Coincidencias web: {len(web_matches)}\n\n"
            f"{'COINCIDENCIAS:' if web_matches else 'No se encontraron copias directas.'}\n"
            + "\n".join(f"- '{m['sentence']}' → {m['web_result']}" for m in web_matches)
            + f"\n\nEVALUA:\n- Originalidad: X/100\n- Riesgo: bajo/medio/alto\n"
            f"- Sugiere reescribir frases similares si las hay"
        )

    return [check_plagiarism]
