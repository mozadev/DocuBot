"""
Personas / Avatares de cliente ideal.
El agente genera contenido especifico para cada persona.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CustomerPersona:
    """Avatar de cliente ideal."""
    id: str
    name: str
    age_range: str
    gender: str
    location: str = ""
    occupation: str = ""
    income_level: str = ""
    interests: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    preferred_channels: List[str] = field(default_factory=list)
    buying_behavior: str = ""
    objections: List[str] = field(default_factory=list)
    language: str = "es"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "age_range": self.age_range,
            "gender": self.gender,
            "location": self.location,
            "occupation": self.occupation,
            "income_level": self.income_level,
            "interests": self.interests,
            "pain_points": self.pain_points,
            "goals": self.goals,
            "preferred_channels": self.preferred_channels,
            "buying_behavior": self.buying_behavior,
            "objections": self.objections,
            "language": self.language,
        }

    def to_agent_prompt(self) -> str:
        """Convierte la persona en instrucciones para el agente."""
        parts = [
            f"PERSONA: {self.name}",
            f"Edad: {self.age_range} | Genero: {self.gender}",
        ]
        if self.location:
            parts.append(f"Ubicacion: {self.location}")
        if self.occupation:
            parts.append(f"Ocupacion: {self.occupation}")
        if self.income_level:
            parts.append(f"Nivel de ingresos: {self.income_level}")
        if self.interests:
            parts.append(f"Intereses: {', '.join(self.interests)}")
        if self.pain_points:
            parts.append(f"Problemas/necesidades: {'; '.join(self.pain_points)}")
        if self.goals:
            parts.append(f"Objetivos: {'; '.join(self.goals)}")
        if self.preferred_channels:
            parts.append(f"Canales preferidos: {', '.join(self.preferred_channels)}")
        if self.buying_behavior:
            parts.append(f"Comportamiento de compra: {self.buying_behavior}")
        if self.objections:
            parts.append(f"Objeciones comunes: {'; '.join(self.objections)}")
        parts.append(
            f"\nINSTRUCCION: Genera contenido que hable DIRECTAMENTE a {self.name}. "
            f"Responde a sus pain points, usa lenguaje que resuene con su perfil, "
            f"y anticipa sus objeciones en el copy."
        )
        return "\n".join(parts)


@dataclass
class PerformanceFeedback:
    """Feedback de rendimiento de una campana/ad anterior.
    NestJS lo envia despues de que Meta devuelve resultados."""
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "ad_title": self.ad_title,
            "channel": self.channel,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "ctr": self.ctr,
            "cpc": self.cpc,
            "conversions": self.conversions,
            "cost_per_conversion": self.cost_per_conversion,
            "spend": self.spend,
            "roas": self.roas,
            "engagement_rate": self.engagement_rate,
            "whatsapp_messages": self.whatsapp_messages,
            "best_audience_segment": self.best_audience_segment,
            "best_time_of_day": self.best_time_of_day,
            "notes": self.notes,
        }

    def to_agent_prompt(self) -> str:
        parts = [f"RENDIMIENTO AD: {self.ad_title or self.campaign_id} ({self.channel})"]
        if self.impressions:
            parts.append(f"  Impresiones: {self.impressions:,}")
        if self.clicks:
            parts.append(f"  Clicks: {self.clicks:,}")
        if self.ctr:
            parts.append(f"  CTR: {self.ctr:.2%}")
        if self.cpc:
            parts.append(f"  CPC: ${self.cpc:.2f} {self.currency}")
        if self.conversions:
            parts.append(f"  Conversiones: {self.conversions}")
        if self.cost_per_conversion:
            parts.append(f"  Costo/conversion: ${self.cost_per_conversion:.2f} {self.currency}")
        if self.roas:
            parts.append(f"  ROAS: {self.roas:.1f}x")
        if self.whatsapp_messages:
            parts.append(f"  Mensajes WhatsApp: {self.whatsapp_messages}")
        if self.best_audience_segment:
            parts.append(f"  Mejor segmento: {self.best_audience_segment}")
        if self.best_time_of_day:
            parts.append(f"  Mejor horario: {self.best_time_of_day}")
        return "\n".join(parts)
