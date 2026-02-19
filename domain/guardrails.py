"""
Guardrails: validacion de seguridad de marca y contenido.
Previene que el agente genere contenido inapropiado, off-brand, o peligroso.
Se ejecuta ANTES de entregar contenido al tenant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from core.logger import logger


@dataclass
class GuardrailResult:
    """Resultado de la validacion de guardrails."""
    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_content: str = ""
    risk_score: float = 0.0  # 0.0 = safe, 1.0 = blocked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "warnings": self.warnings,
            "risk_score": self.risk_score,
        }


# Palabras/frases que siempre deben ser bloqueadas en contenido de marketing
BLOCKED_PATTERNS = [
    r"\b(garantizado|100%\s*seguro|cura\s+definitiva|milagro)\b",
    r"\b(gratis|free)\b.*\b(sin\s+condiciones|sin\s+letra\s+chica)\b",
    r"\b(bitcoin|crypto|forex|trading)\b.*\b(ganancias|profit|rendimiento)\b",
    r"\b(adelgaza|baja\s+de\s+peso)\b.*\b(sin\s+esfuerzo|sin\s+dieta|rapido)\b",
    r"\b(hack|exploit|pirate)\b",
]

# Patrones que generan advertencia pero no bloqueo
WARNING_PATTERNS = [
    (r"\b(mejor|number\s*one|#1|numero\s*1|lider)\b", "Claim de superioridad puede requerir sustento legal"),
    (r"\b(oferta|descuento|promo)\b.*\b(tiempo\s+limitado|ultimas?\s+unidades?|se\s+acaba)\b",
     "Urgencia artificial — verificar que sea real"),
    (r"\b(doctor|medico|salud|tratamiento)\b", "Contenido de salud requiere disclaimers legales"),
    (r"\b(inversion|rendimiento|ganancias)\b", "Contenido financiero requiere disclaimers legales"),
]

# Requisitos legales por tipo de contenido
LEGAL_REQUIREMENTS = {
    "health": "Incluir: 'Consulte a su medico'. No prometer curas.",
    "finance": "Incluir: 'Rendimientos pasados no garantizan resultados futuros'.",
    "food": "Si menciona propiedades saludables, debe tener sustento.",
    "alcohol": "Incluir: 'El abuso del alcohol es danino'. Solo mayores de edad.",
    "supplements": "Incluir: 'Este producto no es un medicamento'.",
}


class ContentGuardrails:
    """Valida contenido de marketing contra reglas de seguridad de marca."""

    def __init__(
        self,
        brand_never_include: Optional[List[str]] = None,
        industry: str = "",
        strict_mode: bool = False,
    ) -> None:
        self._brand_blocked = [p.lower() for p in (brand_never_include or [])]
        self._industry = industry.lower()
        self._strict = strict_mode

    def validate(self, content: str) -> GuardrailResult:
        """Ejecuta todas las validaciones sobre el contenido."""
        violations: List[str] = []
        warnings: List[str] = []

        content_lower = content.lower()

        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                violations.append(f"Contenido bloqueado: patron '{pattern}' detectado")

        for pattern, warning_msg in WARNING_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                warnings.append(warning_msg)

        for blocked_word in self._brand_blocked:
            if blocked_word.lower() in content_lower:
                violations.append(f"Palabra prohibida por la marca: '{blocked_word}'")

        for industry_key, disclaimer in LEGAL_REQUIREMENTS.items():
            if industry_key in self._industry or industry_key in content_lower:
                if not any(d_word in content_lower for d_word in ["consulte", "disclaimer", "condiciones"]):
                    warnings.append(f"Posible requisito legal: {disclaimer}")

        if len(content) > 5000:
            warnings.append("Contenido muy largo (>5000 chars). Considerar acortar.")

        if self._has_excessive_caps(content):
            warnings.append("Uso excesivo de mayusculas. Puede parecer spam.")

        if self._has_excessive_emojis(content):
            warnings.append("Demasiados emojis. Puede reducir credibilidad.")

        risk_score = self._calculate_risk(violations, warnings)
        passed = len(violations) == 0 and (not self._strict or len(warnings) == 0)

        if violations:
            logger.warning(f"Guardrails: {len(violations)} violaciones en contenido")

        return GuardrailResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            sanitized_content=content if passed else "",
            risk_score=risk_score,
        )

    def validate_campaign(self, campaign_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Valida una campana completa (strategy + content pieces)."""
        results = {"overall_passed": True, "pieces": []}

        strategy = campaign_dict.get("strategy_summary", "")
        if strategy:
            sr = self.validate(strategy)
            if not sr.passed:
                results["overall_passed"] = False
            results["strategy_check"] = sr.to_dict()

        for i, piece in enumerate(campaign_dict.get("content_pieces", [])):
            body = piece.get("body", "")
            title = piece.get("title", "")
            full_content = f"{title}\n{body}"

            pr = self.validate(full_content)
            if not pr.passed:
                results["overall_passed"] = False

            results["pieces"].append({
                "index": i,
                "channel": piece.get("channel", ""),
                "title": title[:50],
                **pr.to_dict(),
            })

        return results

    @staticmethod
    def _has_excessive_caps(text: str) -> bool:
        words = text.split()
        if len(words) < 5:
            return False
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
        return caps_words / len(words) > 0.3

    @staticmethod
    def _has_excessive_emojis(text: str) -> bool:
        emoji_count = sum(1 for c in text if ord(c) > 0x1F600)
        word_count = max(len(text.split()), 1)
        return emoji_count / word_count > 0.15

    @staticmethod
    def _calculate_risk(violations: List[str], warnings: List[str]) -> float:
        score = len(violations) * 0.4 + len(warnings) * 0.1
        return min(score, 1.0)
