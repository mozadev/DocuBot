"""
Entidades del dominio de DocuBot AI.
Modelos puros sin dependencias de frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DocumentChunk:
    """Un fragmento de documento listo para indexar."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return self.metadata.get("filename", "Desconocido")

    @property
    def content_type(self) -> str:
        return self.metadata.get("content_type", "text")

    @property
    def is_image(self) -> bool:
        return self.content_type == "image"


@dataclass
class SearchResult:
    """Resultado de una búsqueda con score de relevancia."""
    chunk: DocumentChunk
    score: float


@dataclass
class Source:
    """Fuente citada en una respuesta."""
    filename: str
    content: str
    score: float
    content_type: str = "text"
    image_path: str = ""
    page_number: int = 0


@dataclass
class ChatResponse:
    """Respuesta del agente al usuario."""
    answer: str
    sources: List[Source] = field(default_factory=list)
    confidence: float = 0.0
    question: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [
                {
                    "filename": s.filename,
                    "content": s.content,
                    "score": s.score,
                    "content_type": s.content_type,
                    "image_path": s.image_path,
                    "page_number": s.page_number,
                }
                for s in self.sources
            ],
            "confidence": self.confidence,
            "question": self.question,
        }


@dataclass
class ChatMessage:
    """Mensaje en el historial de conversación."""
    role: str  # "human" | "ai"
    content: str


@dataclass
class MCPServerInfo:
    """Info de un servidor MCP para la UI."""
    name: str
    description: str
    enabled: bool
    command: str


@dataclass
class MCPStatus:
    """Estado general de MCP."""
    initialized: bool = False
    servers: List[MCPServerInfo] = field(default_factory=list)
    connected_servers: int = 0
    tool_count: int = 0
    tool_names: List[str] = field(default_factory=list)


# ---- Marketing: Contexto de Negocio (viene de NestJS) ----

@dataclass
class ProductInfo:
    """Producto/servicio del tenant."""
    name: str
    description: str = ""
    price: float = 0.0
    currency: str = "USD"
    category: str = ""
    image_url: str = ""
    is_top_seller: bool = False

@dataclass
class WhatsAppMetrics:
    """Metricas de WhatsApp que NestJS ya tiene."""
    total_conversations: int = 0
    avg_daily_messages: int = 0
    top_questions: List[str] = field(default_factory=list)
    peak_hours: List[int] = field(default_factory=list)
    avg_response_time_seconds: int = 0
    conversion_rate: float = 0.0

@dataclass
class SalesData:
    """Datos de ventas del tenant."""
    total_sales_last_30d: float = 0.0
    total_orders_last_30d: int = 0
    avg_ticket: float = 0.0
    top_products: List[str] = field(default_factory=list)
    currency: str = "USD"

@dataclass
class PreviousAdPerformance:
    """Rendimiento de ads anteriores (si NestJS los tiene de Meta Insights)."""
    avg_cpc: float = 0.0
    avg_ctr: float = 0.0
    avg_cpm: float = 0.0
    best_performing_ad: str = ""
    best_audience_segment: str = ""
    total_spend_last_30d: float = 0.0
    total_conversions_last_30d: int = 0
    currency: str = "USD"

@dataclass
class BusinessContext:
    """Contexto completo del negocio — NestJS lo arma y se lo pasa a DocuBot.
    Todos los campos son opcionales; DocuBot usa lo que reciba."""
    business_name: str = ""
    industry: str = ""
    location: str = ""
    products: List[ProductInfo] = field(default_factory=list)
    whatsapp_metrics: Optional[WhatsAppMetrics] = None
    sales_data: Optional[SalesData] = None
    previous_ads: Optional[PreviousAdPerformance] = None
    competitor_names: List[str] = field(default_factory=list)
    brand_colors: List[str] = field(default_factory=list)
    brand_voice: str = ""


# ---- Marketing: Requests y Responses ----

@dataclass
class CampaignRequest:
    """Solicitud de generacion de campana."""
    tenant_id: str
    business_description: str
    target_audience: str = ""
    channels: List[str] = field(default_factory=lambda: ["facebook", "instagram"])
    budget_range: str = ""
    goals: str = ""
    tone: str = "profesional"
    language: str = "es"
    business_context: Optional[BusinessContext] = None

@dataclass
class ContentPiece:
    """Pieza de contenido generada para una campana."""
    channel: str
    content_type: str  # "post", "email", "ad_copy", "story", "caption"
    title: str
    body: str
    hashtags: List[str] = field(default_factory=list)
    call_to_action: str = ""
    suggested_image_prompt: str = ""
    target_audience_detail: str = ""
    suggested_budget_daily: float = 0.0
    suggested_duration_days: int = 7
    placement: List[str] = field(default_factory=list)

@dataclass
class CampaignResponse:
    """Respuesta con la campana generada."""
    campaign_name: str
    strategy_summary: str
    content_pieces: List[ContentPiece] = field(default_factory=list)
    calendar_suggestion: str = ""
    kpi_suggestions: List[str] = field(default_factory=list)
    estimated_reach: str = ""
    audience_recommendation: str = ""
    budget_recommendation: str = ""
    data_insights: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_name": self.campaign_name,
            "strategy_summary": self.strategy_summary,
            "content_pieces": [
                {
                    "channel": cp.channel,
                    "content_type": cp.content_type,
                    "title": cp.title,
                    "body": cp.body,
                    "hashtags": cp.hashtags,
                    "call_to_action": cp.call_to_action,
                    "suggested_image_prompt": cp.suggested_image_prompt,
                    "target_audience_detail": cp.target_audience_detail,
                    "suggested_budget_daily": cp.suggested_budget_daily,
                    "suggested_duration_days": cp.suggested_duration_days,
                    "placement": cp.placement,
                }
                for cp in self.content_pieces
            ],
            "calendar_suggestion": self.calendar_suggestion,
            "kpi_suggestions": self.kpi_suggestions,
            "estimated_reach": self.estimated_reach,
            "audience_recommendation": self.audience_recommendation,
            "budget_recommendation": self.budget_recommendation,
            "data_insights": self.data_insights,
        }


@dataclass
class ContentRequest:
    """Solicitud de contenido especifico."""
    tenant_id: str
    content_type: str  # "social_post", "email", "whatsapp", "ad", "blog"
    topic: str
    tone: str = "profesional"
    max_length: int = 500
    include_hashtags: bool = True
    include_cta: bool = True
    language: str = "es"
    business_context: Optional[BusinessContext] = None


@dataclass
class BrandMemory:
    """Memoria de marca del tenant — el agente lo recuerda siempre."""
    tenant_id: str
    brand_name: str = ""
    brand_voice: str = ""
    brand_colors: List[str] = field(default_factory=list)
    always_include: List[str] = field(default_factory=list)
    never_include: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    target_persona: str = ""
    unique_selling_points: List[str] = field(default_factory=list)
    competitor_differentiators: List[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        parts = []
        if self.brand_name:
            parts.append(f"MARCA: {self.brand_name}")
        if self.brand_voice:
            parts.append(f"VOZ DE MARCA: {self.brand_voice}")
        if self.brand_colors:
            parts.append(f"COLORES: {', '.join(self.brand_colors)}")
        if self.always_include:
            parts.append(f"SIEMPRE INCLUIR: {'; '.join(self.always_include)}")
        if self.never_include:
            parts.append(f"NUNCA INCLUIR: {'; '.join(self.never_include)}")
        if self.key_phrases:
            parts.append(f"FRASES CLAVE: {'; '.join(self.key_phrases)}")
        if self.target_persona:
            parts.append(f"PERSONA OBJETIVO: {self.target_persona}")
        if self.unique_selling_points:
            parts.append(f"DIFERENCIADORES: {'; '.join(self.unique_selling_points)}")
        return "\n".join(parts) if parts else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "brand_name": self.brand_name,
            "brand_voice": self.brand_voice,
            "brand_colors": self.brand_colors,
            "always_include": self.always_include,
            "never_include": self.never_include,
            "key_phrases": self.key_phrases,
            "target_persona": self.target_persona,
            "unique_selling_points": self.unique_selling_points,
            "competitor_differentiators": self.competitor_differentiators,
        }
