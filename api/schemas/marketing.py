"""Schemas de marketing: campanas, contenido, templates, personas, performance."""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from api.schemas.common import BusinessContextSchema


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
    prompt: str = Field(..., min_length=10)
    channel: str = Field("instagram_feed")
    brand_colors: List[str] = Field(default_factory=list)
    quality: str = Field("standard")


class BrandMemorySchema(BaseModel):
    brand_name: str = ""
    brand_voice: str = ""
    brand_colors: List[str] = Field(default_factory=list)
    always_include: List[str] = Field(default_factory=list)
    never_include: List[str] = Field(default_factory=list)
    key_phrases: List[str] = Field(default_factory=list)
    target_persona: str = ""
    unique_selling_points: List[str] = Field(default_factory=list)
    competitor_differentiators: List[str] = Field(default_factory=list)


class TemplatedCampaignRequest(BaseModel):
    template_id: str = Field(...)
    business_description: str = Field(..., min_length=10)
    target_audience: str = ""
    budget_range: str = ""
    goals: str = ""
    language: str = "es"
    business_context: Optional[BusinessContextSchema] = None


class PersonaSchema(BaseModel):
    name: str = Field(...)
    age_range: str = Field(...)
    gender: str = Field(...)
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
    what_to_improve: str = Field("CTR")
    business_context: Optional[BusinessContextSchema] = None


class TranslateRequest(BaseModel):
    content: str = Field(..., min_length=10)
    target_languages: List[str] = Field(...)
    adapt_culturally: bool = True


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None


class ChatResponseSchema(BaseModel):
    answer: str
    sources: list
    confidence: float
    question: str
