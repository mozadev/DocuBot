"""
DocuBot AI — FastAPI REST API.
Diseñada para ser consumida por NestJS, Flutter, React o cualquier cliente HTTP.
Soporta multi-tenancy via header X-Tenant-ID.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.factory import create_all_services
from config.settings import settings
from domain.models import (
    CampaignRequest, ContentRequest, BrandMemory,
    BusinessContext, ProductInfo, WhatsAppMetrics,
    SalesData, PreviousAdPerformance,
)
from domain.guardrails import ContentGuardrails
from adapters.cache.semantic_cache import SemanticCache
from adapters.observability.tracer import AgentTracer, SpanType
from api.rate_limiter import RateLimitMiddleware, get_rate_limiter
from core.logger import logger


# ──────────────────────────── Pydantic Schemas ────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None


class ChatResponseSchema(BaseModel):
    answer: str
    sources: list
    confidence: float
    question: str


# -- Schemas de contexto de negocio (NestJS los arma con datos de su BD) --

class ProductInfoSchema(BaseModel):
    name: str
    description: str = ""
    price: float = 0.0
    currency: str = "USD"
    category: str = ""
    image_url: str = ""
    is_top_seller: bool = False

class WhatsAppMetricsSchema(BaseModel):
    total_conversations: int = 0
    avg_daily_messages: int = 0
    top_questions: List[str] = Field(default_factory=list)
    peak_hours: List[int] = Field(default_factory=list, description="Horas 0-23 con mas actividad")
    avg_response_time_seconds: int = 0
    conversion_rate: float = Field(0.0, description="0.0 a 1.0")

class SalesDataSchema(BaseModel):
    total_sales_last_30d: float = 0.0
    total_orders_last_30d: int = 0
    avg_ticket: float = 0.0
    top_products: List[str] = Field(default_factory=list)
    currency: str = "USD"

class PreviousAdPerformanceSchema(BaseModel):
    avg_cpc: float = Field(0.0, description="Costo por click promedio")
    avg_ctr: float = Field(0.0, description="Click-through rate 0.0 a 1.0")
    avg_cpm: float = Field(0.0, description="Costo por mil impresiones")
    best_performing_ad: str = ""
    best_audience_segment: str = ""
    total_spend_last_30d: float = 0.0
    total_conversions_last_30d: int = 0
    currency: str = "USD"

class BusinessContextSchema(BaseModel):
    """Contexto de negocio que NestJS arma con datos de su BD.
    Todos los campos son opcionales — DocuBot usa lo que reciba."""
    business_name: str = ""
    industry: str = ""
    location: str = ""
    products: List[ProductInfoSchema] = Field(default_factory=list)
    whatsapp_metrics: Optional[WhatsAppMetricsSchema] = None
    sales_data: Optional[SalesDataSchema] = None
    previous_ads: Optional[PreviousAdPerformanceSchema] = None
    competitor_names: List[str] = Field(default_factory=list)
    brand_colors: List[str] = Field(default_factory=list)
    brand_voice: str = ""


# -- Schemas de marketing --

class CampaignRequestSchema(BaseModel):
    business_description: str = Field(..., min_length=10)
    target_audience: str = ""
    channels: List[str] = Field(default=["facebook", "instagram"])
    budget_range: str = ""
    goals: str = ""
    tone: str = "profesional"
    language: str = "es"
    business_context: Optional[BusinessContextSchema] = None

class ContentRequestSchema(BaseModel):
    content_type: str = Field(..., description="social_post | email | whatsapp | ad | blog")
    topic: str = Field(..., min_length=5)
    tone: str = "profesional"
    max_length: int = 500
    include_hashtags: bool = True
    include_cta: bool = True
    language: str = "es"
    business_context: Optional[BusinessContextSchema] = None

class MarketAnalysisRequest(BaseModel):
    query: str = Field(..., min_length=5)
    business_context: Optional[BusinessContextSchema] = None

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=10, description="Descripcion de la imagen a generar")
    channel: str = Field("instagram_feed", description="instagram_feed | instagram_story | facebook_feed | ad_landscape")
    brand_colors: List[str] = Field(default_factory=list, description="Colores de marca hex, ej: ['#FF6B9D']")
    quality: str = Field("standard", description="standard | hd")

class BrandMemorySchema(BaseModel):
    brand_name: str = ""
    brand_voice: str = Field("", description="Ej: 'Empoderada, energetica, cercana'")
    brand_colors: List[str] = Field(default_factory=list)
    always_include: List[str] = Field(default_factory=list, description="Frases/elementos que siempre deben aparecer")
    never_include: List[str] = Field(default_factory=list, description="Palabras/temas prohibidos")
    key_phrases: List[str] = Field(default_factory=list, description="Slogans o frases de la marca")
    target_persona: str = Field("", description="Descripcion del cliente ideal")
    unique_selling_points: List[str] = Field(default_factory=list)
    competitor_differentiators: List[str] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class StatusResponse(BaseModel):
    service: str
    version: str
    model: str
    embedding_model: str
    multimodal: bool
    vision_model: str
    document_count: int
    mcp: dict


# ──────────────────────────── App State ────────────────────────────

_services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa servicios al arrancar, limpia al apagar."""
    logger.info("FastAPI: inicializando servicios...")
    doc_svc, chat_svc, mkt_svc, dalle = create_all_services(session_id="fastapi")
    _services["doc"] = doc_svc
    _services["chat"] = chat_svc
    _services["mkt"] = mkt_svc
    _services["dalle"] = dalle
    _services["cache"] = SemanticCache(max_entries=1000, default_ttl=3600)
    _services["tracer"] = AgentTracer(max_traces=500)
    _services["guardrails"] = ContentGuardrails()

    from api.streaming import create_streaming_routes
    from api.webhooks import create_webhook_routes
    app.include_router(create_streaming_routes(_services))
    app.include_router(create_webhook_routes(_services))

    logger.info("FastAPI: servicios listos (con cache, tracer, guardrails, streaming, webhooks)")
    yield
    _services.clear()
    logger.info("FastAPI: servicios liberados")


def _get_tenant_session(tenant_id: str, base_session: str = "") -> str:
    return f"{tenant_id}_{base_session}" if tenant_id else base_session


# ──────────────────────────── FastAPI Instance ────────────────────────────

app = FastAPI(
    title="DocuBot AI API",
    description=(
        "API REST de DocuBot AI — Agente LangGraph + RAG Multimodal + MCP + Marketing.\n\n"
        "Diseñada para integrarse con backends NestJS, clientes Flutter/React, etc.\n"
        "Soporta multi-tenancy mediante el header `X-Tenant-ID`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────── Health ────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(status="ok", service="DocuBot AI", version="1.0.0")


@app.get("/api/v1/status", response_model=StatusResponse, tags=["Health"])
async def status():
    doc_svc = _services.get("doc")
    chat_svc = _services.get("chat")
    mcp_info = chat_svc.get_mcp_status() if chat_svc else None

    return StatusResponse(
        service="DocuBot AI",
        version="1.0.0",
        model=settings.openai_model,
        embedding_model=settings.embedding_model,
        multimodal=settings.enable_multimodal,
        vision_model=settings.vision_model,
        document_count=doc_svc.get_document_count() if doc_svc else 0,
        mcp={
            "initialized": mcp_info.initialized if mcp_info else False,
            "connected_servers": mcp_info.connected_servers if mcp_info else 0,
            "tools": mcp_info.tool_names if mcp_info else [],
        },
    )


# ──────────────────────────── Documents ────────────────────────────

@app.post("/api/v1/documents/upload", tags=["Documents"])
async def upload_documents(
    files: List[UploadFile] = File(...),
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Sube y procesa documentos (PDF, DOCX). Retorna stats de indexación."""
    doc_svc = _services["doc"]
    results = []

    for f in files:
        ext = f"'.{f.filename.split('.')[-1].lower()}" if f.filename else ""
        if ext.lstrip("'") not in [e for e in doc_svc.supported_extensions]:
            results.append({"filename": f.filename, "error": f"Formato no soportado: {ext}"})
            continue

        try:
            suffix = f".{f.filename.split('.')[-1].lower()}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await f.read()
                tmp.write(content)
                tmp_path = tmp.name

            stats = doc_svc.process_and_index(tmp_path)
            os.unlink(tmp_path)
            results.append({
                "filename": f.filename,
                "text_chunks": stats["text_chunks"],
                "image_chunks": stats["image_chunks"],
                "total_chunks": stats["total"],
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    total_indexed = sum(r.get("total_chunks", 0) for r in results)
    return {
        "tenant_id": x_tenant_id,
        "files_processed": len(results),
        "total_chunks_indexed": total_indexed,
        "details": results,
    }


@app.get("/api/v1/documents/stats", tags=["Documents"])
async def document_stats(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
    doc_svc = _services["doc"]
    count = doc_svc.get_document_count()
    return {
        "tenant_id": x_tenant_id,
        "total_chunks": count,
        "status": "active" if count > 0 else "empty",
        "supported_formats": doc_svc.supported_extensions,
    }


@app.delete("/api/v1/documents", tags=["Documents"])
async def clear_documents(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
    doc_svc = _services["doc"]
    doc_svc.clear_database()
    return {"tenant_id": x_tenant_id, "message": "Base de datos limpiada"}


# ──────────────────────────── Chat ────────────────────────────

@app.post("/api/v1/chat", response_model=ChatResponseSchema, tags=["Chat"])
async def chat(
    body: ChatRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Envía una pregunta al agente LangGraph + RAG + MCP."""
    chat_svc = _services["chat"]
    response = chat_svc.ask_question(body.question)
    return ChatResponseSchema(
        answer=response.answer,
        sources=[s.__dict__ if hasattr(s, "__dict__") else s for s in response.sources],
        confidence=response.confidence,
        question=response.question,
    )


@app.get("/api/v1/chat/history", tags=["Chat"])
async def chat_history(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
    chat_svc = _services["chat"]
    history = chat_svc.get_chat_history()
    return {
        "tenant_id": x_tenant_id,
        "messages": [{"role": m.role, "content": m.content} for m in history],
    }


@app.post("/api/v1/chat/summary", tags=["Chat"])
async def chat_summary(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
    chat_svc = _services["chat"]
    summary = chat_svc.get_conversation_summary()
    return {"tenant_id": x_tenant_id, "summary": summary}


@app.delete("/api/v1/chat/memory", tags=["Chat"])
async def clear_memory(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
    chat_svc = _services["chat"]
    chat_svc.clear_memory()
    return {"tenant_id": x_tenant_id, "message": "Memoria del chat limpiada"}


# ──────────────────────────── Marketing ────────────────────────────

def _schema_to_business_context(s: Optional[BusinessContextSchema]) -> Optional[BusinessContext]:
    """Convierte el schema Pydantic de la API al dataclass del dominio."""
    if not s:
        return None
    products = [
        ProductInfo(
            name=p.name, description=p.description, price=p.price,
            currency=p.currency, category=p.category,
            image_url=p.image_url, is_top_seller=p.is_top_seller,
        )
        for p in (s.products or [])
    ]
    wa = None
    if s.whatsapp_metrics:
        wm = s.whatsapp_metrics
        wa = WhatsAppMetrics(
            total_conversations=wm.total_conversations,
            avg_daily_messages=wm.avg_daily_messages,
            top_questions=wm.top_questions,
            peak_hours=wm.peak_hours,
            avg_response_time_seconds=wm.avg_response_time_seconds,
            conversion_rate=wm.conversion_rate,
        )
    sales = None
    if s.sales_data:
        sd = s.sales_data
        sales = SalesData(
            total_sales_last_30d=sd.total_sales_last_30d,
            total_orders_last_30d=sd.total_orders_last_30d,
            avg_ticket=sd.avg_ticket,
            top_products=sd.top_products,
            currency=sd.currency,
        )
    prev_ads = None
    if s.previous_ads:
        pa = s.previous_ads
        prev_ads = PreviousAdPerformance(
            avg_cpc=pa.avg_cpc, avg_ctr=pa.avg_ctr, avg_cpm=pa.avg_cpm,
            best_performing_ad=pa.best_performing_ad,
            best_audience_segment=pa.best_audience_segment,
            total_spend_last_30d=pa.total_spend_last_30d,
            total_conversions_last_30d=pa.total_conversions_last_30d,
            currency=pa.currency,
        )
    return BusinessContext(
        business_name=s.business_name,
        industry=s.industry,
        location=s.location,
        products=products,
        whatsapp_metrics=wa,
        sales_data=sales,
        previous_ads=prev_ads,
        competitor_names=s.competitor_names or [],
        brand_colors=s.brand_colors or [],
        brand_voice=s.brand_voice,
    )


@app.post("/api/v1/marketing/campaign", tags=["Marketing"])
async def generate_campaign(
    body: CampaignRequestSchema,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Genera una campana de marketing digital completa.
    Si NestJS envia business_context con datos reales, la campana sera data-driven."""
    mkt_svc = _services["mkt"]

    request = CampaignRequest(
        tenant_id=x_tenant_id,
        business_description=body.business_description,
        target_audience=body.target_audience,
        channels=body.channels,
        budget_range=body.budget_range,
        goals=body.goals,
        tone=body.tone,
        language=body.language,
        business_context=_schema_to_business_context(body.business_context),
    )

    campaign = mkt_svc.generate_campaign(request)
    return {
        "tenant_id": x_tenant_id,
        "data_driven": request.business_context is not None,
        "campaign": campaign.to_dict(),
    }


@app.post("/api/v1/marketing/content", tags=["Marketing"])
async def generate_content(
    body: ContentRequestSchema,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Genera una pieza individual de contenido de marketing."""
    mkt_svc = _services["mkt"]

    request = ContentRequest(
        tenant_id=x_tenant_id,
        content_type=body.content_type,
        topic=body.topic,
        tone=body.tone,
        max_length=body.max_length,
        include_hashtags=body.include_hashtags,
        include_cta=body.include_cta,
        language=body.language,
        business_context=_schema_to_business_context(body.business_context),
    )

    piece = mkt_svc.generate_content(request)
    return {
        "tenant_id": x_tenant_id,
        "data_driven": request.business_context is not None,
        "content": {
            "channel": piece.channel,
            "content_type": piece.content_type,
            "title": piece.title,
            "body": piece.body,
            "hashtags": piece.hashtags,
            "call_to_action": piece.call_to_action,
            "suggested_image_prompt": piece.suggested_image_prompt,
        },
    }


@app.post("/api/v1/marketing/analyze", tags=["Marketing"])
async def analyze_market(
    body: MarketAnalysisRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Analiza datos de mercado/marketing con datos reales del negocio."""
    mkt_svc = _services["mkt"]
    biz_ctx = _schema_to_business_context(body.business_context)
    analysis = mkt_svc.analyze_market(body.query, business_context=biz_ctx)
    return {
        "tenant_id": x_tenant_id,
        "data_driven": biz_ctx is not None,
        "analysis": analysis,
    }


# ──────────────────────────── Image Generation ────────────────────────────

@app.post("/api/v1/marketing/image", tags=["Marketing"])
async def generate_image(
    body: ImageGenerationRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Genera una imagen publicitaria con DALL-E 3."""
    dalle = _services.get("dalle")
    if not dalle:
        raise HTTPException(status_code=503, detail="DALL-E no configurado")

    result = dalle.generate_ad_image(
        product_description=body.prompt,
        brand_colors=body.brand_colors or None,
        channel=body.channel,
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return {
        "tenant_id": x_tenant_id,
        "images": result["images"],
    }


# ──────────────────────────── Brand Memory ────────────────────────────

_brand_memories: dict = {}


@app.put("/api/v1/marketing/brand-memory", tags=["Marketing"])
async def save_brand_memory(
    body: BrandMemorySchema,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Guarda la memoria de marca del tenant. El agente la usara en todas las generaciones."""
    memory = BrandMemory(
        tenant_id=x_tenant_id,
        brand_name=body.brand_name,
        brand_voice=body.brand_voice,
        brand_colors=body.brand_colors,
        always_include=body.always_include,
        never_include=body.never_include,
        key_phrases=body.key_phrases,
        target_persona=body.target_persona,
        unique_selling_points=body.unique_selling_points,
        competitor_differentiators=body.competitor_differentiators,
    )
    _brand_memories[x_tenant_id] = memory
    return {"tenant_id": x_tenant_id, "message": "Brand memory guardada", "brand": memory.to_dict()}


@app.get("/api/v1/marketing/brand-memory", tags=["Marketing"])
async def get_brand_memory(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
    """Obtiene la memoria de marca del tenant."""
    memory = _brand_memories.get(x_tenant_id)
    if not memory:
        return {"tenant_id": x_tenant_id, "brand": None, "message": "Sin brand memory configurada"}
    return {"tenant_id": x_tenant_id, "brand": memory.to_dict()}


# ──────────────────────────── Templates ────────────────────────────

@app.get("/api/v1/marketing/templates", tags=["Templates"])
async def list_campaign_templates():
    """Lista todos los templates de campana disponibles por industria."""
    from domain.templates import list_templates
    return {"templates": list_templates()}


@app.get("/api/v1/marketing/templates/{template_id}", tags=["Templates"])
async def get_campaign_template(template_id: str):
    """Obtiene un template completo con todos los detalles."""
    from domain.templates import get_template
    template = get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' no encontrado")
    return {"template": template.to_dict()}


class TemplatedCampaignRequest(BaseModel):
    template_id: str = Field(..., description="ID del template (restaurant, gym_fitness, etc.)")
    business_description: str = Field(..., min_length=10)
    target_audience: str = ""
    budget_range: str = ""
    goals: str = ""
    language: str = "es"
    business_context: Optional[BusinessContextSchema] = None


@app.post("/api/v1/marketing/campaign-from-template", tags=["Templates"])
async def generate_campaign_from_template(
    body: TemplatedCampaignRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Genera una campana basada en un template de industria + datos reales del tenant."""
    from domain.templates import get_template
    template = get_template(body.template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{body.template_id}' no encontrado")

    mkt_svc = _services["mkt"]
    enhanced_description = (
        f"{body.business_description}\n\n"
        f"USA ESTE TEMPLATE COMO BASE:\n{template.to_agent_prompt()}"
    )

    request = CampaignRequest(
        tenant_id=x_tenant_id,
        business_description=enhanced_description,
        target_audience=body.target_audience or template.audience_hints,
        channels=template.suggested_channels,
        budget_range=body.budget_range or template.suggested_budget_range,
        goals=body.goals or template.suggested_objective,
        tone=template.tone,
        language=body.language,
        business_context=_schema_to_business_context(body.business_context),
    )

    campaign = mkt_svc.generate_campaign(request)
    return {
        "tenant_id": x_tenant_id,
        "template_used": template.id,
        "data_driven": request.business_context is not None,
        "campaign": campaign.to_dict(),
    }


# ──────────────────────────── Personas ────────────────────────────

class PersonaSchema(BaseModel):
    name: str = Field(..., description="Nombre del avatar, ej: 'Maria Fitness'")
    age_range: str = Field(..., description="Ej: '25-35'")
    gender: str = Field(..., description="Ej: 'Mujer'")
    location: str = ""
    occupation: str = ""
    income_level: str = ""
    interests: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    preferred_channels: List[str] = Field(default_factory=list)
    buying_behavior: str = ""
    objections: List[str] = Field(default_factory=list)

class PersonaCampaignRequest(BaseModel):
    persona: PersonaSchema
    business_description: str = Field(..., min_length=10)
    channels: List[str] = Field(default=["instagram", "facebook"])
    budget_range: str = ""
    goals: str = ""
    tone: str = "profesional"
    language: str = "es"
    business_context: Optional[BusinessContextSchema] = None


@app.post("/api/v1/marketing/campaign-for-persona", tags=["Personas"])
async def generate_campaign_for_persona(
    body: PersonaCampaignRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Genera una campana personalizada para un avatar/persona especifico."""
    from domain.personas import CustomerPersona

    persona = CustomerPersona(
        id=body.persona.name.lower().replace(" ", "_"),
        name=body.persona.name,
        age_range=body.persona.age_range,
        gender=body.persona.gender,
        location=body.persona.location,
        occupation=body.persona.occupation,
        income_level=body.persona.income_level,
        interests=body.persona.interests,
        pain_points=body.persona.pain_points,
        goals=body.persona.goals,
        preferred_channels=body.persona.preferred_channels,
        buying_behavior=body.persona.buying_behavior,
        objections=body.persona.objections,
    )

    mkt_svc = _services["mkt"]
    enhanced_description = (
        f"{body.business_description}\n\n"
        f"GENERA CONTENIDO PARA ESTA PERSONA:\n{persona.to_agent_prompt()}"
    )

    request = CampaignRequest(
        tenant_id=x_tenant_id,
        business_description=enhanced_description,
        target_audience=f"{persona.name}: {persona.gender} {persona.age_range}, {persona.location}",
        channels=body.channels,
        budget_range=body.budget_range,
        goals=body.goals,
        tone=body.tone,
        language=body.language,
        business_context=_schema_to_business_context(body.business_context),
    )

    campaign = mkt_svc.generate_campaign(request)
    return {
        "tenant_id": x_tenant_id,
        "persona": persona.to_dict(),
        "campaign": campaign.to_dict(),
    }


# ──────────────────────────── Performance Learning ────────────────────────────

class PerformanceFeedbackSchema(BaseModel):
    campaign_id: str
    ad_title: str = ""
    channel: str = ""
    impressions: int = 0
    clicks: int = 0
    ctr: float = 0.0
    cpc: float = 0.0
    conversions: int = 0
    cost_per_conversion: float = 0.0
    spend: float = 0.0
    roas: float = 0.0
    engagement_rate: float = 0.0
    whatsapp_messages: int = 0
    best_audience_segment: str = ""
    best_time_of_day: str = ""
    notes: str = ""
    currency: str = "USD"

class OptimizationRequest(BaseModel):
    performance_history: List[PerformanceFeedbackSchema]
    business_description: str = Field(..., min_length=10)
    what_to_improve: str = Field("CTR", description="KPI a mejorar: CTR, CPC, conversions, ROAS")
    business_context: Optional[BusinessContextSchema] = None


_performance_history: Dict[str, list] = {}


@app.post("/api/v1/marketing/performance-feedback", tags=["Performance"])
async def submit_performance_feedback(
    body: List[PerformanceFeedbackSchema],
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """NestJS envia resultados de Meta Ads. DocuBot aprende para futuras campanas."""
    from domain.personas import PerformanceFeedback

    feedbacks = []
    for fb in body:
        pf = PerformanceFeedback(
            campaign_id=fb.campaign_id, ad_title=fb.ad_title, channel=fb.channel,
            impressions=fb.impressions, clicks=fb.clicks, ctr=fb.ctr, cpc=fb.cpc,
            conversions=fb.conversions, cost_per_conversion=fb.cost_per_conversion,
            spend=fb.spend, roas=fb.roas, engagement_rate=fb.engagement_rate,
            whatsapp_messages=fb.whatsapp_messages,
            best_audience_segment=fb.best_audience_segment,
            best_time_of_day=fb.best_time_of_day, notes=fb.notes, currency=fb.currency,
        )
        feedbacks.append(pf)

    if x_tenant_id not in _performance_history:
        _performance_history[x_tenant_id] = []
    _performance_history[x_tenant_id].extend(feedbacks)

    return {
        "tenant_id": x_tenant_id,
        "feedbacks_received": len(feedbacks),
        "total_history": len(_performance_history[x_tenant_id]),
        "message": "Performance feedback guardado. Se usara en futuras campanas.",
    }


@app.post("/api/v1/marketing/optimize", tags=["Performance"])
async def suggest_campaign_optimization(
    body: OptimizationRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Analiza rendimiento y sugiere optimizaciones basadas en datos reales."""
    from domain.personas import PerformanceFeedback

    perf_text = "\n\n".join(
        PerformanceFeedback(
            campaign_id=fb.campaign_id, ad_title=fb.ad_title, channel=fb.channel,
            impressions=fb.impressions, clicks=fb.clicks, ctr=fb.ctr, cpc=fb.cpc,
            conversions=fb.conversions, cost_per_conversion=fb.cost_per_conversion,
            spend=fb.spend, roas=fb.roas, engagement_rate=fb.engagement_rate,
            whatsapp_messages=fb.whatsapp_messages,
            best_audience_segment=fb.best_audience_segment,
            best_time_of_day=fb.best_time_of_day, notes=fb.notes, currency=fb.currency,
        ).to_agent_prompt()
        for fb in body.performance_history
    )

    mkt_svc = _services["mkt"]
    biz_ctx = _schema_to_business_context(body.business_context)
    analysis = mkt_svc.analyze_market(
        f"Optimiza la siguiente campana para mejorar {body.what_to_improve}:\n\n"
        f"Negocio: {body.business_description}\n\n"
        f"RENDIMIENTO ACTUAL:\n{perf_text}\n\n"
        f"Genera recomendaciones concretas y accionables.",
        business_context=biz_ctx,
    )

    return {
        "tenant_id": x_tenant_id,
        "kpi_to_improve": body.what_to_improve,
        "optimization_suggestions": analysis,
    }


# ──────────────────────────── Multi-idioma ────────────────────────────

class TranslateRequest(BaseModel):
    content: str = Field(..., min_length=10)
    target_languages: List[str] = Field(..., description="Codigos de idioma: en, pt, fr, de, it, etc.")
    adapt_culturally: bool = Field(True, description="Adaptar culturalmente vs traduccion literal")


@app.post("/api/v1/marketing/translate", tags=["Multi-idioma"])
async def translate_content(
    body: TranslateRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Traduce y adapta culturalmente contenido de marketing a multiples idiomas."""
    mkt_svc = _services["mkt"]
    translations = {}

    lang_names = {
        "en": "Ingles (USA)", "pt": "Portugues (Brasil)", "fr": "Frances",
        "de": "Aleman", "it": "Italiano", "ja": "Japones", "zh": "Chino Mandarín",
        "ko": "Coreano", "ar": "Arabe", "hi": "Hindi",
    }

    for lang in body.target_languages:
        lang_name = lang_names.get(lang, lang)
        adapt = "adaptacion cultural completa" if body.adapt_culturally else "traduccion literal"

        result = mkt_svc._llm.invoke([
            {"role": "system", "content": (
                f"Eres un traductor experto en marketing. Traduce al {lang_name} con {adapt}. "
                f"Adapta modismos, hashtags y CTAs al mercado del idioma destino. "
                f"Responde SOLO con la traduccion, sin explicaciones."
            )},
            {"role": "user", "content": body.content},
        ])
        translations[lang] = {"language": lang_name, "content": result}

    return {
        "tenant_id": x_tenant_id,
        "original_language": "es",
        "translations": translations,
    }


# ──────────────────────────── SEO ────────────────────────────

class SEOKeywordRequest(BaseModel):
    business_type: str = Field(..., min_length=3)
    location: str = ""
    language: str = "es"


class SEOBlogRequest(BaseModel):
    topic: str = Field(..., min_length=5)
    primary_keyword: str = Field(..., min_length=2)
    secondary_keywords: List[str] = Field(default_factory=list)
    word_count: int = Field(1500, ge=500, le=5000)
    tone: str = "profesional"
    business_context: Optional[BusinessContextSchema] = None


class SEOMetaRequest(BaseModel):
    page_title: str = Field(..., min_length=3)
    page_description: str = Field(..., min_length=10)
    primary_keyword: str = Field(..., min_length=2)


class SEOScoreRequest(BaseModel):
    content: str = Field(..., min_length=50)
    target_keyword: str = Field(..., min_length=2)


@app.post("/api/v1/seo/keywords", tags=["SEO"])
async def research_keywords(
    body: SEOKeywordRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Investiga keywords SEO: primarias, secundarias, long-tail con volumen estimado."""
    mkt_svc = _services["mkt"]
    result = mkt_svc._llm.invoke([
        {"role": "system", "content": (
            "Eres un experto SEO. Investiga keywords para este negocio. "
            "Retorna una tabla con keyword | volumen_estimado | dificultad | intento_busqueda. "
            "Incluye: 5 primarias, 10 secundarias, 15 long-tail."
        )},
        {"role": "user", "content": (
            f"Negocio: {body.business_type}\n"
            f"Ubicacion: {body.location}\n"
            f"Idioma: {body.language}"
        )},
    ])
    return {"tenant_id": x_tenant_id, "keywords": result}


@app.post("/api/v1/seo/blog", tags=["SEO"])
async def generate_seo_blog(
    body: SEOBlogRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Genera un blog post completo optimizado para SEO con H1, H2, meta description, FAQ."""
    mkt_svc = _services["mkt"]

    rag_context = ""
    biz_ctx = None
    if body.business_context:
        biz_ctx = _schema_to_business_context(body.business_context)
    try:
        results = _services["doc"].search(body.topic, k=3)
        rag_context = "\n".join(r.chunk.content[:200] for r in results)
    except Exception:
        pass

    secondary = ", ".join(body.secondary_keywords) if body.secondary_keywords else ""

    result = mkt_svc._llm.invoke([
        {"role": "system", "content": (
            "Eres un content writer SEO experto. Genera blog posts que rankean en Google. "
            "ESTRUCTURA: H1 (keyword, <60 chars) → Meta description (150-160 chars) → "
            "Intro con keyword → 4-6 H2s → FAQ section → Conclusion con CTA. "
            "Densidad keyword: 1-2%. Oraciones cortas. Incluye schema markup suggestion."
        )},
        {"role": "user", "content": (
            f"Tema: {body.topic}\n"
            f"Keyword primaria: {body.primary_keyword}\n"
            f"Keywords secundarias: {secondary}\n"
            f"Palabras: ~{body.word_count}\n"
            f"Tono: {body.tone}\n"
            f"Contexto del negocio: {rag_context[:500]}"
        )},
    ])
    return {"tenant_id": x_tenant_id, "blog_post": result, "keyword": body.primary_keyword}


@app.post("/api/v1/seo/meta-tags", tags=["SEO"])
async def generate_meta_tags(
    body: SEOMetaRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Genera meta tags SEO: title, description, Open Graph, Twitter Cards, Schema.org."""
    mkt_svc = _services["mkt"]
    result = mkt_svc._llm.invoke([
        {"role": "system", "content": (
            "Genera meta tags SEO optimizados. Incluye: "
            "1) <title> (<60 chars, keyword al inicio) "
            "2) <meta description> (150-160 chars) "
            "3) Open Graph tags "
            "4) Twitter Card tags "
            "5) Schema.org JSON-LD. "
            "Formato: HTML listo para copiar."
        )},
        {"role": "user", "content": (
            f"Pagina: {body.page_title}\n"
            f"Descripcion: {body.page_description}\n"
            f"Keyword: {body.primary_keyword}"
        )},
    ])
    return {"tenant_id": x_tenant_id, "meta_tags": result}


@app.post("/api/v1/seo/score", tags=["SEO"])
async def seo_score(
    body: SEOScoreRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Analiza contenido existente y calcula SEO score 0-100 con recomendaciones."""
    word_count = len(body.content.split())
    keyword_count = body.content.lower().count(body.target_keyword.lower())
    density = (keyword_count / max(word_count, 1)) * 100

    mkt_svc = _services["mkt"]
    result = mkt_svc._llm.invoke([
        {"role": "system", "content": (
            "Eres un auditor SEO. Califica el contenido de 0 a 100. "
            "Categorias: Titulo /20, Densidad keyword /15, Estructura /20, "
            "Longitud /15, Readability /15, Links /15. "
            "Da un score total y 5 recomendaciones accionables."
        )},
        {"role": "user", "content": (
            f"Keyword: {body.target_keyword}\n"
            f"Palabras: {word_count}\n"
            f"Densidad: {density:.1f}%\n"
            f"Contenido:\n{body.content[:3000]}"
        )},
    ])
    return {
        "tenant_id": x_tenant_id,
        "quick_stats": {
            "word_count": word_count,
            "keyword_count": keyword_count,
            "keyword_density_pct": round(density, 1),
        },
        "analysis": result,
    }


# ──────────────────────────── Plagiarism Check ────────────────────────────

class PlagiarismRequest(BaseModel):
    content: str = Field(..., min_length=20)
    content_type: str = Field("ad_copy", pattern="^(ad_copy|blog_post|email|social_post)$")


@app.post("/api/v1/content/plagiarism-check", tags=["Content Quality"])
async def plagiarism_check(
    body: PlagiarismRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Verifica originalidad del contenido buscando frases en web."""
    from adapters.search.tavily_adapter import TavilyAdapter

    sentences = [s.strip() for s in body.content.replace("\n", ". ").split(". ") if len(s.strip()) > 20]
    check_sentences = sentences[:5]

    tavily: TavilyAdapter = _services.get("doc")
    try:
        from api.factory import _cache as factory_cache
        adapters = factory_cache.get("adapters", {})
        tavily_adapter = adapters.get("tavily")
    except Exception:
        tavily_adapter = None

    matches = []
    if tavily_adapter and tavily_adapter._api_key:
        for sentence in check_sentences[:3]:
            try:
                results = tavily_adapter.search(f'"{sentence}"')
                if results and "no results" not in results.lower() and "no encontr" not in results.lower():
                    matches.append({"sentence": sentence[:100], "found": True, "preview": results[:200]})
                else:
                    matches.append({"sentence": sentence[:100], "found": False})
            except Exception:
                matches.append({"sentence": sentence[:100], "found": False, "error": "search_failed"})
    else:
        matches = [{"sentence": s[:100], "found": False, "note": "Web search not configured"} for s in check_sentences]

    found_count = sum(1 for m in matches if m.get("found"))
    total_checked = len(matches)
    originality = round((1 - found_count / max(total_checked, 1)) * 100)

    return {
        "tenant_id": x_tenant_id,
        "content_type": body.content_type,
        "sentences_checked": total_checked,
        "potential_matches": found_count,
        "originality_score": originality,
        "risk_level": "alto" if found_count >= 2 else ("medio" if found_count == 1 else "bajo"),
        "details": matches,
    }


# ──────────────────────────── Guardrails ────────────────────────────

class GuardrailCheckRequest(BaseModel):
    content: str = Field(..., min_length=1)
    industry: str = ""
    brand_never_include: List[str] = Field(default_factory=list)
    strict_mode: bool = False


@app.post("/api/v1/guardrails/check", tags=["Guardrails"])
async def check_content_safety(
    body: GuardrailCheckRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Valida contenido contra reglas de seguridad de marca antes de publicar."""
    guardrails = ContentGuardrails(
        brand_never_include=body.brand_never_include,
        industry=body.industry,
        strict_mode=body.strict_mode,
    )
    result = guardrails.validate(body.content)
    return {
        "tenant_id": x_tenant_id,
        **result.to_dict(),
        "sanitized_content": result.sanitized_content,
    }


class CampaignGuardrailRequest(BaseModel):
    campaign: dict = Field(..., description="Campana completa con strategy_summary y content_pieces")
    industry: str = ""
    brand_never_include: List[str] = Field(default_factory=list)


@app.post("/api/v1/guardrails/check-campaign", tags=["Guardrails"])
async def check_campaign_safety(
    body: CampaignGuardrailRequest,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Valida una campana completa contra reglas de seguridad de marca."""
    guardrails = ContentGuardrails(
        brand_never_include=body.brand_never_include,
        industry=body.industry,
    )
    result = guardrails.validate_campaign(body.campaign)
    return {"tenant_id": x_tenant_id, **result}


# ──────────────────────────── Cache ────────────────────────────

@app.get("/api/v1/cache/stats", tags=["Cache"])
async def cache_stats(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
    """Estadisticas del cache semantico — hit rate, ahorro estimado."""
    cache: SemanticCache = _services.get("cache")
    if not cache:
        return {"error": "Cache not initialized"}
    return {"tenant_id": x_tenant_id, **cache.get_stats()}


@app.post("/api/v1/cache/invalidate", tags=["Cache"])
async def invalidate_cache(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
    """Invalida cache de un tenant (util cuando sube nuevos documentos)."""
    cache: SemanticCache = _services.get("cache")
    if not cache:
        return {"error": "Cache not initialized"}
    removed = cache.invalidate(x_tenant_id)
    return {"tenant_id": x_tenant_id, "entries_removed": removed}


# ──────────────────────────── Observability ────────────────────────────

@app.get("/api/v1/observability/traces", tags=["Observability"])
async def list_traces(
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
    limit: int = Query(20, ge=1, le=100),
):
    """Lista los ultimos traces de un tenant."""
    tracer: AgentTracer = _services.get("tracer")
    if not tracer:
        return {"error": "Tracer not initialized"}
    return {"tenant_id": x_tenant_id, "traces": tracer.get_tenant_traces(x_tenant_id, limit)}


@app.get("/api/v1/observability/traces/{trace_id}", tags=["Observability"])
async def get_trace_detail(
    trace_id: str,
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Detalle completo de un trace con todos sus spans."""
    tracer: AgentTracer = _services.get("tracer")
    if not tracer:
        return {"error": "Tracer not initialized"}
    trace = tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@app.get("/api/v1/observability/analytics", tags=["Observability"])
async def get_analytics(
    x_tenant_id: str = Header("default", alias="X-Tenant-ID"),
):
    """Analiticas de uso: costos, tokens, latencia, tools mas usadas."""
    tracer: AgentTracer = _services.get("tracer")
    if not tracer:
        return {"error": "Tracer not initialized"}
    return {"tenant_id": x_tenant_id, **tracer.get_analytics(x_tenant_id)}


# ──────────────────────────── Rate Limiting ────────────────────────────

@app.get("/api/v1/rate-limit/usage", tags=["Rate Limiting"])
async def rate_limit_usage(x_tenant_id: str = Header("default", alias="X-Tenant-ID")):
    """Muestra el uso actual del tenant vs sus limites."""
    return get_rate_limiter().get_usage(x_tenant_id)


class SetTierRequest(BaseModel):
    tenant_id: str
    tier: str = Field(..., pattern="^(free|pro|enterprise)$")


@app.post("/api/v1/rate-limit/set-tier", tags=["Rate Limiting"])
async def set_tenant_tier(body: SetTierRequest):
    """NestJS llama esto cuando un tenant cambia de plan."""
    get_rate_limiter().set_tenant_tier(body.tenant_id, body.tier)
    return {"tenant_id": body.tenant_id, "tier": body.tier, "message": "Tier updated"}
