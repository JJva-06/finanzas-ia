"""Piezas visuales de la estética "Boleta térmica": HTML del ticket y estilo de Plotly.

Todo texto que llega al HTML se escapa: las descripciones vienen del archivo del usuario y
los insights del LLM, y ninguno de los dos debe poder inyectar HTML en la página.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import plotly.graph_objects as go

CSS = Path(__file__).resolve().parents[1] / "assets" / "boleta.css"

PAPEL = "#f3eee2"
TINTA = "#1c1a17"
TINTA_GASTADA = "#6e675b"
SELLO = "#c3122f"
FUENTE = "Azeret Mono, ui-monospace, monospace"

# Series de la evolución: color y además estilo de línea, para no depender solo del color.
SERIES = {
    "Ingreso": {"color": TINTA, "dash": "solid"},
    "Gasto": {"color": SELLO, "dash": "solid"},
    "Balance": {"color": TINTA_GASTADA, "dash": "dot"},
}


def hoja_de_estilos() -> str:
    return f"<style>{CSS.read_text(encoding='utf-8')}</style>"


def cabecera() -> str:
    return (
        '<div class="cabecera">'
        '<p class="marca">Mis finanzas</p>'
        '<p class="sub">Asistente con IA · categoriza · resume · responde</p>'
        "</div>"
    )


def corte(doble: bool = False) -> str:
    return f'<hr class="corte{" doble" if doble else ""}">'


def titulo_mes(texto: str) -> str:
    return f'<p class="mes-titulo">{escape(texto)}</p>'


def lineas_kpi(filas: list[tuple[str, str]], total: tuple[str, str], sello: str | None) -> str:
    """Líneas de recibo con puntos guía; la última es el total. El sello va rotado en rojo."""
    html = ['<div class="recibo">']
    for i, (etiqueta, valor) in enumerate(filas):
        html.append(_linea(etiqueta, valor, i))
    html.append(corte())
    html.append(_linea(*total, len(filas), clase="total"))
    if sello:
        html.append(f'<div class="sello-fila"><span class="sello">{escape(sello)}</span></div>')
    html.append("</div>")
    return "".join(html)


def notas(textos: list[str]) -> str:
    items = "".join(f'<li style="--i:{i}">{escape(t)}</li>' for i, t in enumerate(textos))
    return f'<ul class="notas">{items}</ul>'


def pie(lineas: list[str]) -> str:
    """Pie del ticket; cada línea ya debe venir escapada si trae datos externos."""
    cuerpo = "<br>".join(lineas)
    return (
        f'<div class="pie">{cuerpo}'
        '<div class="codigo-barras" aria-hidden="true"></div>'
        "Gracias por su preferencia</div>"
    )


def estilizar(figura: go.Figure) -> go.Figure:
    """Fondo transparente sobre el papel, tipografía del ticket y grilla punteada recesiva."""
    figura.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FUENTE, "color": TINTA, "size": 12},
        hoverlabel={"bgcolor": PAPEL, "bordercolor": TINTA, "font": {"family": FUENTE}},
        legend={"orientation": "h", "y": -0.18, "title_text": None},
    )
    grilla = {"gridcolor": "rgba(28,26,23,0.14)", "griddash": "dot", "zeroline": False}
    figura.update_xaxes(**grilla, linecolor=TINTA)
    figura.update_yaxes(**grilla, linecolor=TINTA)
    return figura


def _linea(etiqueta: str, valor: str, indice: int, clase: str = "") -> str:
    return (
        f'<div class="linea {clase}" style="--i:{indice}">'
        f'<span class="etiqueta">{escape(etiqueta)}</span>'
        '<span class="puntos"></span>'
        f'<span class="valor">{escape(valor)}</span></div>'
    )
