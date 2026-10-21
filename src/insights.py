"""Agregaciones para el dashboard. Todos los números salen de aquí; el LLM solo redacta.

Espera un DataFrame ya categorizado (`categorize.categorizar`). El signo decide los KPIs:
lo positivo es ingreso y lo negativo es gasto, sin importar la categoría.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv

from src.schemas import Categoria

# Respaldo para archivos subidos con dólares; el archivo de ejemplo está solo en soles.
TIPO_CAMBIO_POR_DEFECTO = 3.75
TOP_COMERCIOS = 5


@dataclass(frozen=True)
class KPIsMes:
    mes: str
    gasto: float
    ingreso: float
    balance: float
    variacion_gasto: float | None  # fracción vs. el mes anterior; None si no hay mes anterior


def tipo_cambio_usd() -> float:
    load_dotenv()
    return float(os.getenv("TIPO_CAMBIO_USD") or TIPO_CAMBIO_POR_DEFECTO)


def agregar_monto_pen(df: pd.DataFrame, tipo_cambio: float) -> pd.DataFrame:
    """Agrega `monto_pen` sin tocar `monto` ni `moneda` (el dato original se conserva)."""
    resultado = df.copy()
    factor = resultado["moneda"].map({"PEN": 1.0, "USD": tipo_cambio})
    resultado["monto_pen"] = (resultado["monto"] * factor).round(2)
    return resultado


def hubo_conversion(df: pd.DataFrame) -> bool:
    return bool((df["moneda"] != "PEN").any())


def meses_disponibles(df: pd.DataFrame) -> list[str]:
    return sorted(df["mes"].unique())


def kpis_mes(df: pd.DataFrame, mes: str) -> KPIsMes:
    gasto, ingreso = _totales(df, mes)
    anterior = str(pd.Period(mes, freq="M") - 1)
    variacion = None
    if anterior in set(df["mes"]):
        gasto_anterior, _ = _totales(df, anterior)
        if gasto_anterior > 0:
            variacion = (gasto - gasto_anterior) / gasto_anterior
    return KPIsMes(mes, gasto, ingreso, round(ingreso - gasto, 2), variacion)


def gasto_por_categoria(df: pd.DataFrame, mes: str) -> pd.Series:
    """Gasto (en positivo) por categoría en el mes, de mayor a menor."""
    cargos = _cargos(df, mes)
    return cargos.groupby("categoria")["gasto"].sum().round(2).sort_values(ascending=False)


def evolucion_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Ingreso, gasto y balance de cada mes, en orden cronológico."""
    filas = []
    for mes in meses_disponibles(df):
        gasto, ingreso = _totales(df, mes)
        filas.append({"mes": mes, "ingreso": ingreso, "gasto": gasto, "balance": ingreso - gasto})
    return pd.DataFrame(filas).set_index("mes").round(2)


def top_comercios(df: pd.DataFrame, mes: str, n: int = TOP_COMERCIOS) -> pd.Series:
    """Comercios con más gasto en el mes. Excluye Transferencias: son personas, no comercios."""
    cargos = _cargos(df, mes)
    cargos = cargos[cargos["categoria"] != Categoria.TRANSFERENCIAS]
    return cargos.groupby("descripcion")["gasto"].sum().round(2).nlargest(n)


def _totales(df: pd.DataFrame, mes: str) -> tuple[float, float]:
    del_mes = df.loc[df["mes"] == mes, "monto_pen"]
    gasto = -del_mes[del_mes < 0].sum()
    ingreso = del_mes[del_mes > 0].sum()
    return round(float(gasto), 2), round(float(ingreso), 2)


def _cargos(df: pd.DataFrame, mes: str) -> pd.DataFrame:
    cargos = df[(df["mes"] == mes) & (df["monto_pen"] < 0)]
    return cargos.assign(gasto=-cargos["monto_pen"])
