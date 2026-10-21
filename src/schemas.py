"""Tipos compartidos entre capas: categorías y modelos Pydantic de la salida del LLM."""

from enum import StrEnum

from pydantic import BaseModel


class Categoria(StrEnum):
    COMIDA = "Comida y delivery"
    SUPERMERCADO = "Supermercado"
    TRANSPORTE = "Transporte"
    SUSCRIPCIONES = "Suscripciones"
    ENTRETENIMIENTO = "Entretenimiento"
    SALUD = "Salud"
    EDUCACION = "Educación"
    SERVICIOS = "Servicios (luz, agua, internet, celular)"
    TRANSFERENCIAS = "Transferencias"
    INGRESOS = "Ingresos"
    OTROS = "Otros"


class ResultadoItem(BaseModel):
    """Una descripción categorizada. `categoria` es un enum: el modelo no puede inventar otra."""

    id: int
    categoria: Categoria
    confianza: float


class RespuestaLote(BaseModel):
    resultados: list[ResultadoItem]
