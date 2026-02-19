"""Marketing endpoints: campaigns, content, images, brand memory, templates, personas, performance, i18n."""

from __future__ import annotations
from typing import Dict, List

from fastapi import APIRouter, Header, HTTPException
from domain.models import CampaignRequest, ContentRequest, BrandMemory
from api.schemas.marketing import (
    CampaignRequestSchema, ContentRequestSchema, MarketAnalysisRequest,
    ImageGenerationRequest, BrandMemorySchema, TemplatedCampaignRequest,
    PersonaCampaignRequest, PerformanceFeedbackSchema, OptimizationRequest,
    TranslateRequest,
)
from api.converters import schema_to_business_context

router = APIRouter(prefix="/api/v1/marketing", tags=["Marketing"])

_brand_memories: dict = {}
_performance_history: Dict[str, list] = {}


def create_marketing_routes(services: dict) -> APIRouter:

    @router.post("/campaign")
    async def generate_campaign(body: CampaignRequestSchema, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        request = CampaignRequest(
            tenant_id=x_tenant_id, business_description=body.business_description,
            target_audience=body.target_audience, channels=body.channels,
            budget_range=body.budget_range, goals=body.goals, tone=body.tone,
            language=body.language, business_context=schema_to_business_context(body.business_context),
        )
        campaign = services["mkt"].generate_campaign(request)
        return {"tenant_id": x_tenant_id, "data_driven": request.business_context is not None, "campaign": campaign.to_dict()}

    @router.post("/content")
    async def generate_content(body: ContentRequestSchema, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        request = ContentRequest(
            tenant_id=x_tenant_id, content_type=body.content_type, topic=body.topic,
            tone=body.tone, max_length=body.max_length, include_hashtags=body.include_hashtags,
            include_cta=body.include_cta, language=body.language,
            business_context=schema_to_business_context(body.business_context),
        )
        piece = services["mkt"].generate_content(request)
        return {"tenant_id": x_tenant_id, "data_driven": request.business_context is not None, "content": {
            "channel": piece.channel, "content_type": piece.content_type, "title": piece.title,
            "body": piece.body, "hashtags": piece.hashtags, "call_to_action": piece.call_to_action,
            "suggested_image_prompt": piece.suggested_image_prompt,
        }}

    @router.post("/analyze")
    async def analyze_market(body: MarketAnalysisRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        biz_ctx = schema_to_business_context(body.business_context)
        analysis = services["mkt"].analyze_market(body.query, business_context=biz_ctx)
        return {"tenant_id": x_tenant_id, "data_driven": biz_ctx is not None, "analysis": analysis}

    @router.post("/image")
    async def generate_image(body: ImageGenerationRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        dalle = services.get("dalle")
        if not dalle:
            raise HTTPException(status_code=503, detail="DALL-E no configurado")
        result = dalle.generate_ad_image(product_description=body.prompt, brand_colors=body.brand_colors or None, channel=body.channel)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return {"tenant_id": x_tenant_id, "images": result["images"]}

    @router.put("/brand-memory")
    async def save_brand_memory(body: BrandMemorySchema, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        memory = BrandMemory(
            tenant_id=x_tenant_id, brand_name=body.brand_name, brand_voice=body.brand_voice,
            brand_colors=body.brand_colors, always_include=body.always_include,
            never_include=body.never_include, key_phrases=body.key_phrases,
            target_persona=body.target_persona, unique_selling_points=body.unique_selling_points,
            competitor_differentiators=body.competitor_differentiators,
        )
        _brand_memories[x_tenant_id] = memory
        return {"tenant_id": x_tenant_id, "message": "Brand memory guardada", "brand": memory.to_dict()}

    @router.get("/brand-memory")
    async def get_brand_memory(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        memory = _brand_memories.get(x_tenant_id)
        if not memory:
            return {"tenant_id": x_tenant_id, "brand": None, "message": "Sin brand memory configurada"}
        return {"tenant_id": x_tenant_id, "brand": memory.to_dict()}

    # Templates
    @router.get("/templates", tags=["Templates"])
    async def list_campaign_templates():
        from domain.templates import list_templates
        return {"templates": list_templates()}

    @router.get("/templates/{template_id}", tags=["Templates"])
    async def get_campaign_template(template_id: str):
        from domain.templates import get_template
        template = get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' no encontrado")
        return {"template": template.to_dict()}

    @router.post("/campaign-from-template", tags=["Templates"])
    async def generate_campaign_from_template(body: TemplatedCampaignRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        from domain.templates import get_template
        template = get_template(body.template_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{body.template_id}' no encontrado")
        enhanced = f"{body.business_description}\n\nUSA ESTE TEMPLATE:\n{template.to_agent_prompt()}"
        request = CampaignRequest(
            tenant_id=x_tenant_id, business_description=enhanced,
            target_audience=body.target_audience or template.audience_hints,
            channels=template.suggested_channels, budget_range=body.budget_range or template.suggested_budget_range,
            goals=body.goals or template.suggested_objective, tone=template.tone,
            language=body.language, business_context=schema_to_business_context(body.business_context),
        )
        campaign = services["mkt"].generate_campaign(request)
        return {"tenant_id": x_tenant_id, "template_used": template.id, "campaign": campaign.to_dict()}

    # Personas
    @router.post("/campaign-for-persona", tags=["Personas"])
    async def generate_campaign_for_persona(body: PersonaCampaignRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        from domain.personas import CustomerPersona
        persona = CustomerPersona(
            id=body.persona.name.lower().replace(" ", "_"), name=body.persona.name,
            age_range=body.persona.age_range, gender=body.persona.gender,
            location=body.persona.location, occupation=body.persona.occupation,
            income_level=body.persona.income_level, interests=body.persona.interests,
            pain_points=body.persona.pain_points, goals=body.persona.goals,
            preferred_channels=body.persona.preferred_channels,
            buying_behavior=body.persona.buying_behavior, objections=body.persona.objections,
        )
        enhanced = f"{body.business_description}\n\nGENERA PARA PERSONA:\n{persona.to_agent_prompt()}"
        request = CampaignRequest(
            tenant_id=x_tenant_id, business_description=enhanced,
            target_audience=f"{persona.name}: {persona.gender} {persona.age_range}, {persona.location}",
            channels=body.channels, budget_range=body.budget_range, goals=body.goals,
            tone=body.tone, language=body.language,
            business_context=schema_to_business_context(body.business_context),
        )
        campaign = services["mkt"].generate_campaign(request)
        return {"tenant_id": x_tenant_id, "persona": persona.to_dict(), "campaign": campaign.to_dict()}

    # Performance
    @router.post("/performance-feedback", tags=["Performance"])
    async def submit_performance_feedback(body: List[PerformanceFeedbackSchema], x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        from domain.personas import PerformanceFeedback
        feedbacks = [
            PerformanceFeedback(
                campaign_id=fb.campaign_id, ad_title=fb.ad_title, channel=fb.channel,
                impressions=fb.impressions, clicks=fb.clicks, ctr=fb.ctr, cpc=fb.cpc,
                conversions=fb.conversions, cost_per_conversion=fb.cost_per_conversion,
                spend=fb.spend, roas=fb.roas, engagement_rate=fb.engagement_rate,
                whatsapp_messages=fb.whatsapp_messages, best_audience_segment=fb.best_audience_segment,
                best_time_of_day=fb.best_time_of_day, notes=fb.notes, currency=fb.currency,
            ) for fb in body
        ]
        if x_tenant_id not in _performance_history:
            _performance_history[x_tenant_id] = []
        _performance_history[x_tenant_id].extend(feedbacks)
        return {"tenant_id": x_tenant_id, "feedbacks_received": len(feedbacks), "total_history": len(_performance_history[x_tenant_id])}

    @router.post("/optimize", tags=["Performance"])
    async def suggest_optimization(body: OptimizationRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        from domain.personas import PerformanceFeedback
        perf_text = "\n\n".join(
            PerformanceFeedback(
                campaign_id=fb.campaign_id, ad_title=fb.ad_title, channel=fb.channel,
                impressions=fb.impressions, clicks=fb.clicks, ctr=fb.ctr, cpc=fb.cpc,
                conversions=fb.conversions, cost_per_conversion=fb.cost_per_conversion,
                spend=fb.spend, roas=fb.roas, engagement_rate=fb.engagement_rate,
                whatsapp_messages=fb.whatsapp_messages, best_audience_segment=fb.best_audience_segment,
                best_time_of_day=fb.best_time_of_day, notes=fb.notes, currency=fb.currency,
            ).to_agent_prompt() for fb in body.performance_history
        )
        biz_ctx = schema_to_business_context(body.business_context)
        analysis = services["mkt"].analyze_market(
            f"Optimiza campana para {body.what_to_improve}:\nNegocio: {body.business_description}\nRENDIMIENTO:\n{perf_text}",
            business_context=biz_ctx,
        )
        return {"tenant_id": x_tenant_id, "kpi_to_improve": body.what_to_improve, "optimization_suggestions": analysis}

    # i18n
    @router.post("/translate", tags=["Multi-idioma"])
    async def translate_content(body: TranslateRequest, x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
        lang_names = {"en": "Ingles (USA)", "pt": "Portugues (Brasil)", "fr": "Frances", "de": "Aleman",
                      "it": "Italiano", "ja": "Japones", "zh": "Chino Mandarin", "ko": "Coreano", "ar": "Arabe", "hi": "Hindi"}
        translations = {}
        for lang in body.target_languages:
            lang_name = lang_names.get(lang, lang)
            adapt = "adaptacion cultural completa" if body.adapt_culturally else "traduccion literal"
            result = services["mkt"]._llm.invoke([
                {"role": "system", "content": f"Traduce al {lang_name} con {adapt}. Adapta modismos y CTAs. Solo la traduccion."},
                {"role": "user", "content": body.content},
            ])
            translations[lang] = {"language": lang_name, "content": result}
        return {"tenant_id": x_tenant_id, "original_language": "es", "translations": translations}

    return router
