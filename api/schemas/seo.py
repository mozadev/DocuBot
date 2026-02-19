"""Schemas de SEO y content quality."""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from api.schemas.common import BusinessContextSchema


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


class PlagiarismRequest(BaseModel):
    content: str = Field(..., min_length=20)
    content_type: str = Field("ad_copy", pattern="^(ad_copy|blog_post|email|social_post)$")


class GuardrailCheckRequest(BaseModel):
    content: str = Field(..., min_length=1)
    industry: str = ""
    brand_never_include: List[str] = Field(default_factory=list)
    strict_mode: bool = False


class CampaignGuardrailRequest(BaseModel):
    campaign: dict = Field(...)
    industry: str = ""
    brand_never_include: List[str] = Field(default_factory=list)


class SetTierRequest(BaseModel):
    tenant_id: str
    tier: str = Field(..., pattern="^(free|pro|enterprise)$")
