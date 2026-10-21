"""Redacción de insights: pandas calcula los hechos, el LLM solo los pone en palabras.

Si el LLM no está disponible, responde mal o usa un número que no estaba en sus hechos,
se usa la plantilla de ese hecho. Así ningún número del dashboard sale del LLM.
"""

from __future__ import annotations

import json
import re

from src.insights import Hecho, redactar_con_plantilla
from src.llm import LLMNoDisponible, ProveedorLLM
from src.schemas import TextosInsights

PROMPT_SISTEMA = """Redactas insights breves en español para el dashboard de finanzas personales
de un estudiante en Perú.

Recibirás una lista JSON de hechos ya calculados. Escribe exactamente una frase por hecho,
en el mismo orden, de máximo 25 palabras, en segunda persona y con tono cercano.

Reglas:
- Usa solo los números que aparecen en los hechos, escritos exactamente como vienen
  (por ejemplo "S/ 1,234.50" o "35%").
- No calcules, redondees ni agregues números nuevos (tampoco diferencias ni porcentajes).
- No des recomendaciones de inversión ni consejos financieros."""

_NUMERO_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def redactar_insights(hechos: list[Hecho], llm: ProveedorLLM | None) -> list[str]:
    plantillas = [redactar_con_plantilla(h) for h in hechos]
    if not hechos or llm is None:
        return plantillas

    contenido = json.dumps([{"tipo": h.tipo, **h.datos} for h in hechos], ensure_ascii=False)
    try:
        respuesta = llm.parse(PROMPT_SISTEMA, contenido, TextosInsights)
    except LLMNoDisponible:
        return plantillas
    if respuesta is None or len(respuesta.textos) != len(hechos):
        return plantillas

    return [
        texto if usa_solo_numeros_del_hecho(texto, hecho) else plantilla
        for texto, hecho, plantilla in zip(respuesta.textos, hechos, plantillas, strict=True)
    ]


def usa_solo_numeros_del_hecho(texto: str, hecho: Hecho) -> bool:
    """¿Cada número del texto aparece en los datos del hecho? Si no, el LLM lo inventó."""
    permitidos = set(_NUMERO_RE.findall(" ".join(hecho.datos.values())))
    return set(_NUMERO_RE.findall(texto)) <= permitidos
