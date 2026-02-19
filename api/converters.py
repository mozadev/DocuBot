"""Conversion de schemas Pydantic a dataclasses del dominio."""

from __future__ import annotations
from typing import Optional

from api.schemas.common import BusinessContextSchema
from domain.models import (
    BusinessContext, ProductInfo, WhatsAppMetrics,
    SalesData, PreviousAdPerformance,
)


def schema_to_business_context(s: Optional[BusinessContextSchema]) -> Optional[BusinessContext]:
    """Convierte el schema Pydantic al dataclass del dominio."""
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
            top_questions=wm.top_questions, peak_hours=wm.peak_hours,
            avg_response_time_seconds=wm.avg_response_time_seconds,
            conversion_rate=wm.conversion_rate,
        )
    sales = None
    if s.sales_data:
        sd = s.sales_data
        sales = SalesData(
            total_sales_last_30d=sd.total_sales_last_30d,
            total_orders_last_30d=sd.total_orders_last_30d,
            avg_ticket=sd.avg_ticket, top_products=sd.top_products,
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
        business_name=s.business_name, industry=s.industry,
        location=s.location, products=products, whatsapp_metrics=wa,
        sales_data=sales, previous_ads=prev_ads,
        competitor_names=s.competitor_names or [],
        brand_colors=s.brand_colors or [], brand_voice=s.brand_voice,
    )
