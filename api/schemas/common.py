"""Schemas compartidos: health, status, business context."""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


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
    """Contexto de negocio que NestJS arma con datos de su BD."""
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
