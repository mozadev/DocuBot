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
    logger.info("FastAPI: servicios listos")
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
