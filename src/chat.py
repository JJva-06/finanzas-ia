"""Chat con tool use: el LLM elige herramientas, el código las ejecuta en pandas.

El LLM nunca ve el CSV ni calcula: solo recibe lo que devuelve cada herramienta, con los
montos ya formateados. Nunca se ejecuta código generado por el modelo (no hay exec/eval):
solo funciones de un diccionario fijo.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.insights import (
    detectar_suscripciones,
    formatear_soles,
    gasto_por_categoria,
    kpis_mes,
    meses_disponibles,
    nombre_mes,
)
from src.llm import ProveedorChat
from src.schemas import Categoria

MAX_HERRAMIENTAS = 5
MAX_FILAS = 10
MAX_INTERCAMBIOS = 10
AVISO_PARCIAL = (
    "Llegué al límite de 5 consultas para esta pregunta, así que la respuesta puede estar "
    "incompleta. Prueba dividiéndola en preguntas más cortas."
)


class HerramientaError(Exception):
    """Argumentos inválidos o sin datos; se devuelve al modelo como tool_result con is_error."""


@dataclass
class RespuestaChat:
    texto: str
    herramientas: list[str] = field(default_factory=list)
    parcial: bool = False


# --- Herramientas (se ejecutan en pandas; devuelven solo lo que la pregunta necesita) ---


def gasto_por_categoria_tool(df: pd.DataFrame, categoria: str, mes: str | None = None) -> dict:
    meses = [_validar_mes(df, mes)] if mes else meses_disponibles(df)
    _validar_categoria(categoria)
    total = sum(gasto_por_categoria(df, m).get(categoria, 0.0) for m in meses)
    cargos = _cargos(df, mes)
    return {
        "categoria": categoria,
        "periodo": _periodo(mes),
        "total": formatear_soles(total),
        "cantidad": int((cargos["categoria"] == categoria).sum()),
    }


def top_gastos(df: pd.DataFrame, n: int, mes: str | None = None) -> dict:
    if mes:
        _validar_mes(df, mes)
    n = max(1, min(int(n), MAX_FILAS))
    mayores = _cargos(df, mes).nlargest(n, "gasto")
    return {"periodo": _periodo(mes), "gastos": _filas(mayores)}


def resumen_mensual(df: pd.DataFrame, mes: str) -> dict:
    _validar_mes(df, mes)
    kpis = kpis_mes(df, mes)
    principales = gasto_por_categoria(df, mes).head(3)
    resumen = {
        "mes": _periodo(mes),
        "gasto": formatear_soles(kpis.gasto),
        "ingreso": formatear_soles(kpis.ingreso),
        "balance": formatear_soles(kpis.balance),
        "categorias_principales": {c: formatear_soles(v) for c, v in principales.items()},
    }
    if kpis.variacion_gasto is not None:
        resumen["variacion_gasto_vs_mes_anterior"] = f"{kpis.variacion_gasto:+.1%}"
    return resumen


def listar_suscripciones(df: pd.DataFrame) -> dict:
    suscripciones = detectar_suscripciones(df)
    return {
        "costo_anual_total": formatear_soles(suscripciones["costo_anual"].sum()),
        "suscripciones": [
            {
                "comercio": fila.descripcion,
                "categoria": fila.categoria,
                "mensual": formatear_soles(fila.monto_mensual),
                "anual_estimado": formatear_soles(fila.costo_anual),
                "detectada_por": fila.origen,
            }
            for fila in suscripciones.itertuples()
        ],
    }


def buscar_transacciones(df: pd.DataFrame, texto: str) -> dict:
    """Resumen de lo que coincide con el texto, sin filas (para eso: listar_transacciones)."""
    coincidencias = _coincidencias(df, texto)
    cargos = coincidencias[coincidencias["monto_pen"] < 0].assign(gasto=lambda d: -d["monto_pen"])
    resumen: dict[str, Any] = {"texto": texto, "coincidencias": len(coincidencias)}
    if not cargos.empty:
        resumen |= {
            "cargos": len(cargos),
            "total_cargos": formatear_soles(cargos["gasto"].sum()),
            "promedio_cargo": formatear_soles(cargos["gasto"].mean()),
            "cargo_maximo": _filas(cargos.nlargest(1, "gasto"))[0],
            "cargo_minimo": _filas(cargos.nsmallest(1, "gasto"))[0],
        }
    abonos = coincidencias[coincidencias["monto_pen"] > 0]
    if not abonos.empty:
        resumen |= {
            "abonos": len(abonos),
            "total_abonos": formatear_soles(abonos["monto_pen"].sum()),
        }
    resumen["mas_reciente"] = _filas(coincidencias.nlargest(1, "fecha"))[0]
    return resumen


def listar_transacciones(df: pd.DataFrame, texto: str, n: int = MAX_FILAS) -> dict:
    coincidencias = _coincidencias(df, texto)
    n = max(1, min(int(n), MAX_FILAS))
    recientes = coincidencias.nlargest(n, "fecha")
    return {
        "transacciones": _filas(recientes),
        "nota": f"se muestran {len(recientes)} de {len(coincidencias)}, las más recientes",
    }


HERRAMIENTAS: dict[str, Callable[..., dict]] = {
    "gasto_por_categoria": gasto_por_categoria_tool,
    "top_gastos": top_gastos,
    "resumen_mensual": resumen_mensual,
    "listar_suscripciones": listar_suscripciones,
    "buscar_transacciones": buscar_transacciones,
    "listar_transacciones": listar_transacciones,
}


def ejecutar_herramienta(df: pd.DataFrame, nombre: str, argumentos: dict) -> dict:
    """Solo ejecuta funciones del diccionario HERRAMIENTAS, nunca código del modelo."""
    funcion = HERRAMIENTAS.get(nombre)
    if funcion is None:
        raise HerramientaError(f"La herramienta '{nombre}' no existe.")
    try:
        return funcion(df, **argumentos)
    except TypeError as error:
        raise HerramientaError(f"Argumentos inválidos para '{nombre}'.") from error


def definiciones_herramientas(df: pd.DataFrame) -> list[dict]:
    """Esquemas JSON para la API. Los meses válidos salen de los datos cargados."""
    mes = {"type": "string", "enum": meses_disponibles(df), "description": "Mes AAAA-MM."}
    categoria = {"type": "string", "enum": [c.value for c in Categoria]}
    n = {"type": "integer", "minimum": 1, "maximum": MAX_FILAS}
    texto = {"type": "string", "description": "Nombre del comercio o parte de la descripción."}
    return [
        _herramienta(
            "gasto_por_categoria",
            "Total gastado en una categoría, en un mes o en todo el periodo si no se da mes.",
            {"categoria": categoria, "mes": mes},
            ["categoria"],
        ),
        _herramienta(
            "top_gastos",
            "Los n cargos más grandes (fecha, comercio, monto), en un mes o en todo el periodo.",
            {"n": n, "mes": mes},
            ["n"],
        ),
        _herramienta(
            "resumen_mensual",
            "Gasto, ingreso, balance, variación vs. el mes anterior y categorías principales.",
            {"mes": mes},
            ["mes"],
        ),
        _herramienta(
            "listar_suscripciones",
            "Suscripciones y cobros recurrentes detectados, con su costo mensual y anual.",
            {},
            [],
        ),
        _herramienta(
            "buscar_transacciones",
            "Resumen de las transacciones cuya descripción contiene el texto: cantidad, total, "
            "promedio, máximo, mínimo y la más reciente. No devuelve la lista completa.",
            {"texto": texto},
            ["texto"],
        ),
        _herramienta(
            "listar_transacciones",
            "Lista hasta 10 transacciones (las más recientes) que contienen el texto. Úsala solo "
            "si la pregunta necesita ver transacciones individuales.",
            {"texto": texto, "n": n},
            ["texto"],
        ),
    ]


# --- Bucle de conversación ---


def prompt_sistema(df: pd.DataFrame) -> str:
    meses = ", ".join(f"{m} ({nombre_mes(m)})" for m in meses_disponibles(df))
    return f"""Eres un asistente que responde preguntas sobre el estado de cuenta del usuario.

Meses con datos: {meses}. Si el usuario no dice el año, se refiere a esos meses.

Reglas:
- Responde solo con datos que devuelvan las herramientas. No calcules ni estimes cifras
  que las herramientas no te dieron.
- Si ninguna herramienta permite responder, dilo claramente: "No puedo responder eso con
  los datos que tengo". No inventes.
- Escribe los montos exactamente como los devuelven las herramientas (formato S/ 1,234.50).
- Responde en español, breve y directo.
- No des recomendaciones de inversión."""


def responder(
    pregunta: str, historial: list[dict], df: pd.DataFrame, llm: ProveedorChat
) -> RespuestaChat:
    """Responde una pregunta y agrega el intercambio (solo texto) al historial.

    Puede lanzar LLMNoDisponible; la UI lo muestra como un mensaje amigable.
    """
    mensajes = historial[-2 * MAX_INTERCAMBIOS :] + [{"role": "user", "content": pregunta}]
    sistema, herramientas = prompt_sistema(df), definiciones_herramientas(df)
    usadas: list[str] = []

    # Cota de vueltas: a lo sumo MAX_HERRAMIENTAS rondas con herramientas + 1 para responder.
    for _ in range(MAX_HERRAMIENTAS + 1):
        limite = len(usadas) >= MAX_HERRAMIENTAS
        respuesta = llm.conversar(
            sistema, mensajes, herramientas, {"type": "none"} if limite else None
        )
        mensajes.append({"role": "assistant", "content": respuesta.content})
        pedidos = [b for b in respuesta.content if b.type == "tool_use"]
        if respuesta.stop_reason != "tool_use" or not pedidos:
            break
        mensajes.append({"role": "user", "content": _ejecutar_pedidos(df, pedidos, usadas)})

    texto = "\n".join(b.text for b in respuesta.content if b.type == "text").strip()
    parcial = len(usadas) >= MAX_HERRAMIENTAS
    if parcial:
        texto = f"{texto}\n\n_{AVISO_PARCIAL}_".strip()
    texto = texto or "No pude generar una respuesta. Intenta reformular la pregunta."

    historial.extend(
        [{"role": "user", "content": pregunta}, {"role": "assistant", "content": texto}]
    )
    return RespuestaChat(texto, usadas, parcial)


def _ejecutar_pedidos(df: pd.DataFrame, pedidos: list[Any], usadas: list[str]) -> list[dict]:
    """Ejecuta los tool_use de una vuelta y devuelve todos los tool_result en un solo mensaje."""
    resultados = []
    for pedido in pedidos:
        if len(usadas) >= MAX_HERRAMIENTAS:
            contenido, es_error = "Límite de consultas alcanzado; responde con lo que tienes.", True
        else:
            usadas.append(pedido.name)
            try:
                contenido = json.dumps(
                    ejecutar_herramienta(df, pedido.name, dict(pedido.input)), ensure_ascii=False
                )
                es_error = False
            except HerramientaError as error:
                contenido, es_error = str(error), True
        resultado = {"type": "tool_result", "tool_use_id": pedido.id, "content": contenido}
        if es_error:
            resultado["is_error"] = True
        resultados.append(resultado)
    return resultados


# --- Auxiliares ---


def _herramienta(nombre: str, descripcion: str, propiedades: dict, requeridos: list[str]) -> dict:
    return {
        "name": nombre,
        "description": descripcion,
        "input_schema": {
            "type": "object",
            "properties": propiedades,
            "required": requeridos,
            "additionalProperties": False,
        },
    }


def _validar_mes(df: pd.DataFrame, mes: str) -> str:
    if mes not in meses_disponibles(df):
        disponibles = ", ".join(meses_disponibles(df))
        raise HerramientaError(f"No hay datos de {mes}. Meses disponibles: {disponibles}.")
    return mes


def _validar_categoria(categoria: str) -> None:
    if categoria not in {c.value for c in Categoria}:
        raise HerramientaError(f"La categoría '{categoria}' no existe.")


def _periodo(mes: str | None) -> str:
    if mes is None:
        return "todos los meses"
    return f"{nombre_mes(mes)} {pd.Period(mes, freq='M').year}"


def _cargos(df: pd.DataFrame, mes: str | None) -> pd.DataFrame:
    filtro = df["monto_pen"] < 0
    if mes is not None:
        filtro &= df["mes"] == mes
    return df[filtro].assign(gasto=lambda d: -d["monto_pen"])


def _coincidencias(df: pd.DataFrame, texto: str) -> pd.DataFrame:
    termino = texto.strip().upper()
    if not termino:
        raise HerramientaError("El texto de búsqueda está vacío.")
    coincidencias = df[df["descripcion"].str.contains(termino, regex=False)]
    if coincidencias.empty:
        raise HerramientaError(f"No hay transacciones que contengan '{texto}'.")
    return coincidencias


def _filas(df: pd.DataFrame) -> list[dict]:
    """Solo fecha, comercio y monto: nunca id, capa, confianza ni la descripción original."""
    return [
        {
            "fecha": fila.fecha.strftime("%d/%m/%Y"),
            "comercio": fila.descripcion,
            "monto": formatear_soles(fila.monto_pen),
        }
        for fila in df.itertuples()
    ]
