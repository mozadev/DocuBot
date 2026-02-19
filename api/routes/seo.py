"""SEO & content quality endpoints."""

from __future__ import annotations
from typing import List

from fastapi import APIRouter, Header
from api.schemas.seo import SEOKeywordRequest, SEOBlogRequest, SEOMetaRequest, SEOScoreRequest, PlagiarismRequest
from api.converters import schema_to_business_context

router = APIRouter(prefix="/api/v1", tags=["SEO"])


def create_seo_routes(services: dict) -> APIRouter:

    @router.post("/seo/keywords")
    async def research_keywords(body: SEOKeywordRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        result = services["mkt"]._llm.invoke([
            {"role": "system", "content": "Eres experto SEO. Retorna tabla: keyword | volumen | dificultad | intento. 5 primarias, 10 secundarias, 15 long-tail."},
            {"role": "user", "content": f"Negocio: {body.business_type}\nUbicacion: {body.location}\nIdioma: {body.language}"},
        ])
        return {"tenant_id": x_tenant_id, "keywords": result}

    @router.post("/seo/blog")
    async def generate_seo_blog(body: SEOBlogRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        rag_context = ""
        try:
            results = services["doc"].search(body.topic, k=3)
            rag_context = "\n".join(r.chunk.content[:200] for r in results)
        except Exception:
            pass
        secondary = ", ".join(body.secondary_keywords) if body.secondary_keywords else ""
        result = services["mkt"]._llm.invoke([
            {"role": "system", "content": "Content writer SEO. H1 → Meta desc → Intro → 4-6 H2s → FAQ → CTA. Densidad keyword 1-2%."},
            {"role": "user", "content": f"Tema: {body.topic}\nKeyword: {body.primary_keyword}\nSecundarias: {secondary}\nPalabras: ~{body.word_count}\nTono: {body.tone}\nContexto: {rag_context[:500]}"},
        ])
        return {"tenant_id": x_tenant_id, "blog_post": result, "keyword": body.primary_keyword}

    @router.post("/seo/meta-tags")
    async def gen_meta_tags(body: SEOMetaRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        result = services["mkt"]._llm.invoke([
            {"role": "system", "content": "Genera meta tags: <title>, <meta description>, OG, Twitter Cards, Schema.org JSON-LD. HTML listo."},
            {"role": "user", "content": f"Pagina: {body.page_title}\nDescripcion: {body.page_description}\nKeyword: {body.primary_keyword}"},
        ])
        return {"tenant_id": x_tenant_id, "meta_tags": result}

    @router.post("/seo/score")
    async def seo_score(body: SEOScoreRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        word_count = len(body.content.split())
        keyword_count = body.content.lower().count(body.target_keyword.lower())
        density = (keyword_count / max(word_count, 1)) * 100
        result = services["mkt"]._llm.invoke([
            {"role": "system", "content": "Auditor SEO. Califica 0-100: Titulo/20, Densidad/15, Estructura/20, Longitud/15, Readability/15, Links/15. 5 recomendaciones."},
            {"role": "user", "content": f"Keyword: {body.target_keyword}\nPalabras: {word_count}\nDensidad: {density:.1f}%\nContenido:\n{body.content[:3000]}"},
        ])
        return {"tenant_id": x_tenant_id, "quick_stats": {"word_count": word_count, "keyword_count": keyword_count, "keyword_density_pct": round(density, 1)}, "analysis": result}

    @router.post("/content/plagiarism-check", tags=["Content Quality"])
    async def plagiarism_check(body: PlagiarismRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        sentences = [s.strip() for s in body.content.replace("\n", ". ").split(". ") if len(s.strip()) > 20][:5]
        try:
            from api.factory import _cache as factory_cache
            tavily_adapter = factory_cache.get("adapters", {}).get("tavily")
        except Exception:
            tavily_adapter = None
        matches = []
        if tavily_adapter and tavily_adapter._api_key:
            for sentence in sentences[:3]:
                try:
                    results = tavily_adapter.search(f'"{sentence}"')
                    found = results and "no results" not in results.lower() and "no encontr" not in results.lower()
                    matches.append({"sentence": sentence[:100], "found": found, **({"preview": results[:200]} if found else {})})
                except Exception:
                    matches.append({"sentence": sentence[:100], "found": False, "error": "search_failed"})
        else:
            matches = [{"sentence": s[:100], "found": False, "note": "Web search not configured"} for s in sentences]
        found_count = sum(1 for m in matches if m.get("found"))
        originality = round((1 - found_count / max(len(matches), 1)) * 100)
        return {
            "tenant_id": x_tenant_id, "content_type": body.content_type, "sentences_checked": len(matches),
            "potential_matches": found_count, "originality_score": originality,
            "risk_level": "alto" if found_count >= 2 else ("medio" if found_count == 1 else "bajo"),
            "details": matches,
        }

    return router
