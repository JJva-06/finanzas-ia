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
TOLERANCIA_MONTO = 0.10
MESES_MINIMOS_SUSCRIPCION = 2
UMBRAL_HORMIGA = 20.0

MESES_ES = (
    "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre"
).split()


@dataclass(frozen=True)
class Hecho:
    """Un dato ya calculado para redactar. Los valores van formateados (p. ej. "S/ 47.55")."""

    tipo: str  # "ahorro", "categoria_crecio", "categoria_principal" o "hormiga"
    datos: dict[str, str]


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
    anterior = _mes_anterior(df, mes)
    variacion = None
    if anterior is not None:
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


def detectar_suscripciones(df: pd.DataFrame) -> pd.DataFrame:
    """Cargos recurrentes por patrón O con categoría Suscripciones (para que no se escape ninguna).

    Patrón: mismo comercio, un solo cargo por mes y monto similar (±10%) en al menos 2 meses
    consecutivos. `origen` dice si la recurrencia se vio en los datos ("patrón") o si solo
    la sugiere la categoría ("categoría"): en ese caso el costo anual no está confirmado.
    """
    columnas = ["descripcion", "categoria", "monto_mensual", "meses", "costo_anual", "origen"]
    cargos = df[df["monto_pen"] < 0].assign(gasto=lambda d: -d["monto_pen"])
    filas = []
    for descripcion, grupo in cargos.groupby("descripcion", sort=False):
        grupo = grupo.sort_values("mes", kind="stable")
        por_patron = grupo.groupby("mes").size().max() == 1 and _es_recurrente(grupo)
        por_categoria = (grupo["categoria"] == Categoria.SUSCRIPCIONES).any()
        if not (por_patron or por_categoria):
            continue
        monto = round(float(grupo["gasto"].iloc[-1]), 2)
        filas.append(
            {
                "descripcion": descripcion,
                "categoria": grupo["categoria"].iloc[-1],
                "monto_mensual": monto,
                "meses": sorted(grupo["mes"].unique()),
                "costo_anual": round(monto * 12, 2),
                "origen": "patrón" if por_patron else "categoría",
            }
        )
    resultado = pd.DataFrame(filas, columns=columnas)
    return resultado.sort_values("costo_anual", ascending=False).reset_index(drop=True)


def _es_recurrente(grupo: pd.DataFrame) -> bool:
    """¿Hay una racha de meses consecutivos con montos dentro de la tolerancia?"""
    meses = [pd.Period(m, freq="M").ordinal for m in grupo["mes"]]
    montos = list(grupo["gasto"])
    racha = 1
    for i in range(1, len(meses)):
        consecutivo = meses[i] - meses[i - 1] == 1
        similar = abs(montos[i] - montos[i - 1]) / montos[i - 1] <= TOLERANCIA_MONTO
        racha = racha + 1 if consecutivo and similar else 1
        if racha >= MESES_MINIMOS_SUSCRIPCION:
            return True
    return False


def formatear_soles(valor: float) -> str:
    signo = "-" if valor < 0 else ""
    return f"{signo}S/ {abs(valor):,.2f}"


def nombre_mes(mes: str) -> str:
    return MESES_ES[pd.Period(mes, freq="M").month - 1]


def hechos_del_mes(df: pd.DataFrame, mes: str) -> list[Hecho]:
    """Los 2 o 3 hechos que se redactan como insights. Solo agregados, nunca transacciones."""
    hechos = [_hecho_ahorro(df, mes), _hecho_categoria(df, mes), _hecho_hormiga(df, mes)]
    return [h for h in hechos if h is not None]


def redactar_con_plantilla(hecho: Hecho) -> str:
    """Versión sin LLM: siempre disponible y sin costo."""
    d = hecho.datos
    if hecho.tipo == "ahorro":
        texto = f"En {d['mes']} tu balance fue de {d['balance']}"
        if "mes_anterior" in d:
            texto += f", frente a {d['balance_anterior']} en {d['mes_anterior']}"
        return texto + "."
    if hecho.tipo == "categoria_crecio":
        texto = (
            f"Tu gasto en {d['categoria']} subió de {d['gasto_anterior']} en {d['mes_anterior']} "
            f"a {d['gasto']} en {d['mes']}"
        )
        if "comercio" in d:
            texto += f"; el mayor fue {d['comercio']} con {d['gasto_comercio']}"
        return texto + "."
    if hecho.tipo == "categoria_principal":
        texto = (
            f"{d['categoria']} fue tu mayor gasto en {d['mes']}: {d['gasto']} ({d['porcentaje']})"
        )
        if "comercio" in d:
            texto += f", sobre todo en {d['comercio']}"
        return texto + "."
    return (
        f"En {d['mes']} hiciste {d['cantidad']} compras de menos de {d['umbral']} "
        f"que suman {d['total']}."
    )


def _hecho_ahorro(df: pd.DataFrame, mes: str) -> Hecho:
    actual = kpis_mes(df, mes)
    datos = {"mes": nombre_mes(mes), "balance": formatear_soles(actual.balance)}
    anterior = _mes_anterior(df, mes)
    if anterior is not None:
        datos["mes_anterior"] = nombre_mes(anterior)
        datos["balance_anterior"] = formatear_soles(kpis_mes(df, anterior).balance)
    return Hecho("ahorro", datos)


def _hecho_categoria(df: pd.DataFrame, mes: str) -> Hecho | None:
    actual = gasto_por_categoria(df, mes)
    if actual.empty:
        return None
    anterior = _mes_anterior(df, mes)
    if anterior is not None:
        previo = gasto_por_categoria(df, anterior).reindex(actual.index, fill_value=0.0)
        aumento = actual - previo
        if aumento.max() > 0:
            categoria = aumento.idxmax()
            datos = {
                "mes": nombre_mes(mes),
                "mes_anterior": nombre_mes(anterior),
                "categoria": categoria,
                "gasto": formatear_soles(actual[categoria]),
                "gasto_anterior": formatear_soles(previo[categoria]),
            }
            return Hecho("categoria_crecio", datos | _comercio_principal(df, mes, categoria))

    categoria = actual.index[0]
    datos = {
        "mes": nombre_mes(mes),
        "categoria": categoria,
        "gasto": formatear_soles(actual[categoria]),
        "porcentaje": f"{actual[categoria] / actual.sum():.0%}",
    }
    return Hecho("categoria_principal", datos | _comercio_principal(df, mes, categoria))


def _hecho_hormiga(df: pd.DataFrame, mes: str) -> Hecho | None:
    cargos = _cargos(df, mes)
    hormiga = cargos[
        (cargos["gasto"] < UMBRAL_HORMIGA) & (cargos["categoria"] != Categoria.TRANSFERENCIAS)
    ]
    if hormiga.empty:
        return None
    datos = {
        "mes": nombre_mes(mes),
        "cantidad": str(len(hormiga)),
        "umbral": formatear_soles(UMBRAL_HORMIGA),
        "total": formatear_soles(hormiga["gasto"].sum()),
    }
    return Hecho("hormiga", datos)


def _comercio_principal(df: pd.DataFrame, mes: str, categoria: str) -> dict[str, str]:
    """El comercio con más gasto de la categoría. Nunca personas (Transferencias)."""
    if categoria == Categoria.TRANSFERENCIAS:
        return {}
    cargos = _cargos(df, mes)
    por_comercio = cargos[cargos["categoria"] == categoria].groupby("descripcion")["gasto"].sum()
    if por_comercio.empty:
        return {}
    comercio = por_comercio.idxmax()
    return {"comercio": comercio, "gasto_comercio": formatear_soles(por_comercio[comercio])}


def _mes_anterior(df: pd.DataFrame, mes: str) -> str | None:
    anterior = str(pd.Period(mes, freq="M") - 1)
    return anterior if anterior in set(df["mes"]) else None


def _totales(df: pd.DataFrame, mes: str) -> tuple[float, float]:
    del_mes = df.loc[df["mes"] == mes, "monto_pen"]
    gasto = -del_mes[del_mes < 0].sum()
    ingreso = del_mes[del_mes > 0].sum()
    return round(float(gasto), 2), round(float(ingreso), 2)


def _cargos(df: pd.DataFrame, mes: str) -> pd.DataFrame:
    cargos = df[(df["mes"] == mes) & (df["monto_pen"] < 0)]
    return cargos.assign(gasto=-cargos["monto_pen"])
