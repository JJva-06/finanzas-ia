"""Tipos compartidos entre capas: categorías y (más adelante) modelos Pydantic del LLM."""

from enum import StrEnum


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
