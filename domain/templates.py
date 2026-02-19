"""
Templates de campanas de marketing por industria.
El tenant elige una industria y el agente personaliza con sus datos reales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CampaignTemplate:
    """Template pre-armado para una industria especifica."""
    id: str
    name: str
    industry: str
    description: str
    suggested_channels: List[str]
    suggested_objective: str
    suggested_budget_range: str
    suggested_duration_days: int
    content_types: List[str]
    audience_hints: str
    tone: str
    key_hooks: List[str]
    cta_suggestions: List[str]
    hashtag_suggestions: List[str]
    image_style: str
    whatsapp_integration: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "industry": self.industry,
            "description": self.description,
            "suggested_channels": self.suggested_channels,
            "suggested_objective": self.suggested_objective,
            "suggested_budget_range": self.suggested_budget_range,
            "suggested_duration_days": self.suggested_duration_days,
            "content_types": self.content_types,
            "audience_hints": self.audience_hints,
            "tone": self.tone,
            "key_hooks": self.key_hooks,
            "cta_suggestions": self.cta_suggestions,
            "hashtag_suggestions": self.hashtag_suggestions,
            "image_style": self.image_style,
            "whatsapp_integration": self.whatsapp_integration,
        }

    def to_agent_prompt(self) -> str:
        """Convierte el template en instrucciones para el agente."""
        return (
            f"TEMPLATE DE INDUSTRIA: {self.industry.upper()}\n"
            f"Nombre: {self.name}\n"
            f"Descripcion: {self.description}\n"
            f"Canales recomendados: {', '.join(self.suggested_channels)}\n"
            f"Objetivo: {self.suggested_objective}\n"
            f"Presupuesto sugerido: {self.suggested_budget_range}\n"
            f"Duracion: {self.suggested_duration_days} dias\n"
            f"Tipos de contenido: {', '.join(self.content_types)}\n"
            f"Audiencia: {self.audience_hints}\n"
            f"Tono: {self.tone}\n"
            f"Hooks efectivos en esta industria:\n"
            + "\n".join(f"  - {h}" for h in self.key_hooks) + "\n"
            f"CTAs recomendados:\n"
            + "\n".join(f"  - {c}" for c in self.cta_suggestions) + "\n"
            f"Hashtags base: {', '.join(self.hashtag_suggestions)}\n"
            f"Estilo de imagen: {self.image_style}\n"
            f"Integracion WhatsApp: {'Si' if self.whatsapp_integration else 'No'}\n\n"
            f"INSTRUCCION: Usa este template como base pero PERSONALIZA completamente "
            f"con los datos reales del negocio del tenant. No uses el template textualmente."
        )


# ──────────────────────────── Templates por Industria ────────────────────────────

TEMPLATES: Dict[str, CampaignTemplate] = {}


def _register(t: CampaignTemplate):
    TEMPLATES[t.id] = t


_register(CampaignTemplate(
    id="restaurant",
    name="Restaurante / Comida",
    industry="Restaurante",
    description="Campana para restaurantes, cafeterias, dark kitchens, food trucks",
    suggested_channels=["instagram", "facebook", "whatsapp"],
    suggested_objective="traffic",
    suggested_budget_range="$5-15 USD/dia",
    suggested_duration_days=14,
    content_types=["reel_caption", "story", "carousel", "ad_copy"],
    audience_hints="Radio 5-10km del local. Edad 20-45. Intereses: comida, restaurantes, delivery",
    tone="Casual, apetitoso, cercano",
    key_hooks=[
        "Foto/video del platillo estrella con efecto 'food porn'",
        "Detras de camaras de la cocina (autenticidad)",
        "Promocion de temporada / dia especial",
        "Combo o descuento para primeros clientes",
    ],
    cta_suggestions=[
        "Pide por WhatsApp",
        "Reserva tu mesa",
        "Ordena ahora con envio gratis",
        "Ver menu completo",
    ],
    hashtag_suggestions=["#FoodPorn", "#Foodie", "#ComidaCasera", "#RestauranteDia"],
    image_style="Fotos de comida con buena iluminacion, close-up, colores calidos, fondo desenfocado",
))

_register(CampaignTemplate(
    id="gym_fitness",
    name="Gimnasio / Fitness",
    industry="Fitness",
    description="Campana para gimnasios, estudios de yoga/pilates, entrenadores personales",
    suggested_channels=["instagram", "tiktok", "facebook"],
    suggested_objective="leads",
    suggested_budget_range="$8-20 USD/dia",
    suggested_duration_days=21,
    content_types=["reel_caption", "story", "ad_copy", "video_script"],
    audience_hints="Radio 5-15km. Edad 18-45. Intereses: fitness, salud, bienestar, deporte",
    tone="Motivacional, energetico, empoderador",
    key_hooks=[
        "Transformaciones antes/despues (con permiso)",
        "Reto de 30 dias gratuito",
        "Primera clase gratis",
        "Rutinas rapidas de 5-10 min (valor gratuito)",
    ],
    cta_suggestions=[
        "Agenda tu clase gratis",
        "Escribenos por WhatsApp",
        "Empieza tu transformacion hoy",
        "Reserva tu lugar",
    ],
    hashtag_suggestions=["#Fitness", "#GymLife", "#Transformacion", "#VidaSaludable"],
    image_style="Personas entrenando, ambiente energetico, colores vibrantes, buena iluminacion",
))

_register(CampaignTemplate(
    id="clothing_store",
    name="Tienda de Ropa / Moda",
    industry="Moda",
    description="Campana para tiendas de ropa, accesorios, zapatos, moda online",
    suggested_channels=["instagram", "facebook", "tiktok"],
    suggested_objective="sales",
    suggested_budget_range="$10-25 USD/dia",
    suggested_duration_days=14,
    content_types=["carousel", "reel_caption", "story", "ad_copy"],
    audience_hints="Mujeres 18-45, hombres 20-40. Intereses: moda, compras, tendencias, estilo",
    tone="Trendy, aspiracional, exclusivo pero accesible",
    key_hooks=[
        "Lookbook / outfits completos",
        "Unboxing / try-on haul",
        "Ofertas flash / descuento por tiempo limitado",
        "Nueva coleccion / edicion limitada",
    ],
    cta_suggestions=[
        "Compra ahora",
        "Ver coleccion completa",
        "Envio gratis hoy",
        "Pide por WhatsApp tu talla",
    ],
    hashtag_suggestions=["#ModaOnline", "#OOTD", "#NuevaColeccion", "#EstiloUnico"],
    image_style="Lookbook profesional, fondos limpios o lifestyle, modelos diversos, buena iluminacion",
))

_register(CampaignTemplate(
    id="beauty_salon",
    name="Salon de Belleza / Estetica",
    industry="Belleza",
    description="Campana para salones de belleza, barber shops, spas, esteticas",
    suggested_channels=["instagram", "facebook", "whatsapp"],
    suggested_objective="leads",
    suggested_budget_range="$5-15 USD/dia",
    suggested_duration_days=14,
    content_types=["reel_caption", "story", "carousel", "ad_copy"],
    audience_hints="Mujeres 20-55 (salon), hombres 18-40 (barberia). Radio 3-8km",
    tone="Glamoroso, profesional, de confianza",
    key_hooks=[
        "Antes/despues de transformaciones",
        "Proceso satisfactorio (oddly satisfying videos)",
        "Promo primera vez / referido",
        "Tendencias de temporada (colores, cortes)",
    ],
    cta_suggestions=[
        "Agenda tu cita por WhatsApp",
        "Reserva con descuento",
        "Primera visita con 20% OFF",
        "Consulta gratis",
    ],
    hashtag_suggestions=["#Belleza", "#SalonDeBelleza", "#Transformacion", "#HairGoals"],
    image_style="Antes/despues, close-ups de resultados, ambiente lujoso del salon, colores suaves",
))

_register(CampaignTemplate(
    id="dental_clinic",
    name="Clinica Dental / Medica",
    industry="Salud",
    description="Campana para clinicas dentales, medicas, dermatologicas, opticas",
    suggested_channels=["facebook", "instagram", "whatsapp"],
    suggested_objective="leads",
    suggested_budget_range="$10-30 USD/dia",
    suggested_duration_days=30,
    content_types=["ad_copy", "carousel", "story", "email"],
    audience_hints="Edad 25-60. Radio 10-20km. Intereses: salud, bienestar, seguros medicos",
    tone="Profesional, confiable, empatico, educativo",
    key_hooks=[
        "Tips de salud gratuitos (valor educativo)",
        "Antes/despues de tratamientos (con permiso)",
        "Primera consulta gratis o con descuento",
        "Financiamiento / planes de pago",
    ],
    cta_suggestions=[
        "Agenda tu consulta gratis",
        "Llama ahora",
        "Escribenos por WhatsApp",
        "Conoce nuestros planes de pago",
    ],
    hashtag_suggestions=["#SaludDental", "#Sonrisa", "#CuidaTuSalud", "#ClinicaDental"],
    image_style="Profesional y limpio, doctores sonrientes, pacientes satisfechos, ambiente moderno",
))

_register(CampaignTemplate(
    id="real_estate",
    name="Inmobiliaria / Bienes Raices",
    industry="Inmobiliaria",
    description="Campana para inmobiliarias, agentes, desarrollos, renta de propiedades",
    suggested_channels=["facebook", "instagram", "whatsapp"],
    suggested_objective="leads",
    suggested_budget_range="$15-50 USD/dia",
    suggested_duration_days=30,
    content_types=["carousel", "ad_copy", "video_script", "email"],
    audience_hints="Edad 28-55. NSE medio-alto. Intereses: inversion, hogar, decoracion, finanzas",
    tone="Aspiracional, profesional, exclusivo",
    key_hooks=[
        "Tours virtuales / video walkthrough",
        "Precios desde $X / mensualidades",
        "Ultimas unidades / preventa",
        "Inversion con retorno garantizado",
    ],
    cta_suggestions=[
        "Agenda tu visita",
        "Solicita informacion por WhatsApp",
        "Descarga el brochure",
        "Calcula tu credito",
    ],
    hashtag_suggestions=["#BienesRaices", "#TuNuevaCasa", "#Inversion", "#Inmobiliaria"],
    image_style="Fotos HDR de propiedades, drone shots, renders 3D, ambientes decorados",
))

_register(CampaignTemplate(
    id="ecommerce",
    name="E-commerce / Tienda Online",
    industry="E-commerce",
    description="Campana para tiendas online, dropshipping, marketplaces",
    suggested_channels=["instagram", "facebook", "tiktok", "email"],
    suggested_objective="sales",
    suggested_budget_range="$10-30 USD/dia",
    suggested_duration_days=14,
    content_types=["carousel", "reel_caption", "ad_copy", "email", "story"],
    audience_hints="Basado en el producto. Generalmente 18-55. Compradores online frecuentes",
    tone="Directo, con urgencia controlada, orientado a beneficios",
    key_hooks=[
        "Envio gratis / envio express",
        "Descuento por primera compra",
        "Resenas de clientes reales",
        "Unboxing / producto en uso",
        "Oferta flash / timer de urgencia",
    ],
    cta_suggestions=[
        "Compra ahora con envio gratis",
        "Usa codigo: PRIMERA20",
        "Agrega al carrito",
        "Pide por WhatsApp",
    ],
    hashtag_suggestions=["#CompraOnline", "#Oferta", "#EnvioGratis", "#DescuentoHoy"],
    image_style="Producto sobre fondo limpio, lifestyle en uso, flat lay, colores de marca",
))

_register(CampaignTemplate(
    id="professional_services",
    name="Servicios Profesionales",
    industry="Servicios",
    description="Campana para abogados, contadores, consultores, coaches, freelancers",
    suggested_channels=["facebook", "instagram", "linkedin", "email"],
    suggested_objective="leads",
    suggested_budget_range="$8-25 USD/dia",
    suggested_duration_days=21,
    content_types=["ad_copy", "carousel", "email", "video_script"],
    audience_hints="Profesionales 28-55. NSE medio-alto. Intereses segun el servicio especifico",
    tone="Experto, confiable, resultados orientados",
    key_hooks=[
        "Caso de exito / testimonio",
        "Consulta gratuita / diagnostico",
        "Datos impactantes del problema que resuelves",
        "Contenido educativo (tips, guias)",
    ],
    cta_suggestions=[
        "Agenda tu consulta gratis",
        "Descarga la guia gratuita",
        "Escribenos por WhatsApp",
        "Conoce nuestros casos de exito",
    ],
    hashtag_suggestions=["#ConsultoriaEmpresarial", "#CreceTuNegocio", "#Expertos"],
    image_style="Profesional en oficina moderna, graficas de resultados, testimonios con foto",
))

_register(CampaignTemplate(
    id="education",
    name="Educacion / Cursos Online",
    industry="Educacion",
    description="Campana para escuelas, cursos online, academias, tutores, bootcamps",
    suggested_channels=["instagram", "facebook", "tiktok", "email"],
    suggested_objective="leads",
    suggested_budget_range="$5-20 USD/dia",
    suggested_duration_days=21,
    content_types=["reel_caption", "ad_copy", "carousel", "email", "video_script"],
    audience_hints="Edad 18-45. Intereses: educacion, desarrollo personal, carrera profesional",
    tone="Inspirador, accesible, autoridad educativa",
    key_hooks=[
        "Mini-leccion gratis (valor inmediato)",
        "Testimonio de egresado exitoso",
        "Oferta de lanzamiento / early bird",
        "Estadisticas de empleabilidad / resultados",
    ],
    cta_suggestions=[
        "Inscribete ahora",
        "Clase gratis este sabado",
        "Descarga el temario",
        "Agenda asesoria por WhatsApp",
    ],
    hashtag_suggestions=["#Educacion", "#CursosOnline", "#Aprende", "#DesarrolloProfesional"],
    image_style="Personas aprendiendo, ambiente digital moderno, graficos de progreso, capturas de plataforma",
))

_register(CampaignTemplate(
    id="events",
    name="Eventos / Entretenimiento",
    industry="Eventos",
    description="Campana para eventos, conciertos, bodas, conferencias, fiestas",
    suggested_channels=["instagram", "facebook", "tiktok", "whatsapp"],
    suggested_objective="traffic",
    suggested_budget_range="$10-40 USD/dia",
    suggested_duration_days=21,
    content_types=["reel_caption", "story", "ad_copy", "video_script"],
    audience_hints="Segun el evento. Generalmente 18-45. Intereses: entretenimiento, musica, networking",
    tone="Emocionante, FOMO, exclusivo, energetico",
    key_hooks=[
        "Countdown / cuenta regresiva",
        "Early bird / primeros boletos con descuento",
        "Lineup reveal / speaker reveal",
        "Recap de eventos anteriores",
    ],
    cta_suggestions=[
        "Compra tus boletos",
        "No te quedes fuera",
        "Reserva tu lugar ahora",
        "Comparte con tus amigos",
    ],
    hashtag_suggestions=["#Evento", "#SaveTheDate", "#NoTeLoPierdas"],
    image_style="Flyers vibrantes, fotos de eventos pasados con energia, speakers/artistas, countdown",
))


def get_template(template_id: str) -> CampaignTemplate | None:
    return TEMPLATES.get(template_id)


def list_templates() -> List[Dict[str, str]]:
    return [
        {"id": t.id, "name": t.name, "industry": t.industry, "description": t.description}
        for t in TEMPLATES.values()
    ]
