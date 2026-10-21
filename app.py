"""Entrada de Streamlit: subir un estado de cuenta (o usar el de ejemplo) → dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.categorize import CacheCategorias, ResumenCategorizacion, categorizar
from src.ingest import IngestError, cargar_csv
from src.insights import (
    agregar_monto_pen,
    detectar_suscripciones,
    evolucion_mensual,
    formatear_soles,
    gasto_por_categoria,
    hechos_del_mes,
    hubo_conversion,
    kpis_mes,
    meses_disponibles,
    nombre_mes,
    tipo_cambio_usd,
    top_comercios,
)
from src.llm import ClienteLLM
from src.narrativa import redactar_insights

ARCHIVO_EJEMPLO = Path("data/sample/estado_cuenta_ejemplo.csv")
# Paleta categórica validada (orden fijo): azul, naranja, aqua.
COLORES = {"ingreso": "#2a78d6", "gasto": "#eb6834", "balance": "#1baf7a"}
COLOR_UNICO = COLORES["ingreso"]

st.set_page_config(page_title="Finanzas con IA", page_icon="💸", layout="wide")


def main() -> None:
    _inicializar_sesion()
    st.title("💸 Asistente de finanzas personales con IA")
    st.caption(
        "Sube tu estado de cuenta y mira tus gastos categorizados, un resumen del mes "
        "y tus suscripciones. Datos de ejemplo 100% sintéticos."
    )
    _seccion_carga()

    datos = st.session_state.datos
    if datos is None:
        st.info("Sube un CSV con columnas `fecha, descripcion, monto` o usa los datos de ejemplo.")
        return

    df: pd.DataFrame = datos["df"]
    _mostrar_avisos(datos["avisos"], df)
    mes = _selector_mes(df)
    _kpis(df, mes)
    _insights(df, mes)
    _linea_capas(datos["resumen"])

    gastos, evolucion, suscripciones, ia, transacciones = st.tabs(
        ["Gastos", "Evolución", "Suscripciones", "Cómo categorizó la IA", "Transacciones"]
    )
    with gastos:
        _tab_gastos(df, mes)
    with evolucion:
        _tab_evolucion(df)
    with suscripciones:
        _tab_suscripciones(df)
    with ia:
        _tab_ia(datos["resumen"])
    with transacciones:
        _tab_transacciones(df, mes)


# --- Estado y carga ---


def _inicializar_sesion() -> None:
    # Todo vive en la sesión del visitante: la caché nunca se comparte entre usuarios.
    estado = st.session_state
    estado.setdefault("cache", CacheCategorias())
    estado.setdefault("llm", ClienteLLM())
    estado.setdefault("datos", None)
    estado.setdefault("insights", {})
    estado.setdefault("mes_idx", 0)
    estado.setdefault("origen", None)


def _seccion_carga() -> None:
    subir, ejemplo = st.columns([3, 1], vertical_alignment="bottom")
    archivo = subir.file_uploader("Sube tu estado de cuenta (CSV)", type="csv")
    usar_ejemplo = ejemplo.button("Usar datos de ejemplo", type="primary", width="stretch")

    if usar_ejemplo:
        _procesar(ARCHIVO_EJEMPLO.read_bytes(), origen="ejemplo")
    elif archivo is not None and st.session_state.origen != archivo.file_id:
        _procesar(archivo.getvalue(), origen=archivo.file_id)


def _procesar(contenido: bytes, origen: str) -> None:
    estado = st.session_state
    try:
        ingesta = cargar_csv(contenido)
    except IngestError as error:
        estado.datos, estado.origen = None, origen
        st.error(f"No pudimos leer el archivo: {error}")
        return

    with st.spinner("Categorizando con reglas, caché e IA…"):
        df, resumen = categorizar(ingesta.df, estado.cache, estado.llm)
    df = agregar_monto_pen(df, tipo_cambio_usd())

    estado.datos = {"df": df, "resumen": resumen, "avisos": ingesta.avisos + resumen.avisos}
    estado.insights = {}
    estado.origen = origen
    estado.mes_idx = len(meses_disponibles(df)) - 1  # al entrar, el mes más reciente


def _mostrar_avisos(avisos: list[str], df: pd.DataFrame) -> None:
    if avisos:
        with st.expander(f"⚠️ {len(avisos)} aviso(s) sobre tu archivo"):
            for aviso in avisos:
                st.write(f"- {aviso}")
    if hubo_conversion(df):
        st.caption(
            f"Montos en US$ convertidos a soles con un tipo de cambio referencial de "
            f"{tipo_cambio_usd():.2f}."
        )


# --- Resumen del mes ---


def _selector_mes(df: pd.DataFrame) -> str:
    meses = meses_disponibles(df)
    indice = min(st.session_state.mes_idx, len(meses) - 1)
    mes = meses[indice]

    anterior, titulo, siguiente = st.columns([1, 6, 1], vertical_alignment="center")
    anterior.button("◀", on_click=_mover_mes, args=(-1,), disabled=indice == 0, width="stretch")
    siguiente.button(
        "▶", on_click=_mover_mes, args=(1,), disabled=indice == len(meses) - 1, width="stretch"
    )
    anio = pd.Period(mes, freq="M").year
    titulo.markdown(
        f"<h3 style='text-align:center;margin:0'>{nombre_mes(mes).capitalize()} {anio}</h3>",
        unsafe_allow_html=True,
    )
    return mes


def _mover_mes(paso: int) -> None:
    st.session_state.mes_idx += paso


def _kpis(df: pd.DataFrame, mes: str) -> None:
    kpis = kpis_mes(df, mes)
    gasto, ingreso, balance = st.columns(3)
    variacion = None
    if kpis.variacion_gasto is not None:
        anterior = nombre_mes(str(pd.Period(mes, freq="M") - 1))
        variacion = f"{kpis.variacion_gasto:+.1%} vs. {anterior}"
    gasto.metric(
        "Gasto del mes", formatear_soles(kpis.gasto), variacion, delta_color="inverse", border=True
    )
    ingreso.metric("Ingreso del mes", formatear_soles(kpis.ingreso), border=True)
    balance.metric("Balance", formatear_soles(kpis.balance), border=True)


def _insights(df: pd.DataFrame, mes: str) -> None:
    estado = st.session_state
    if mes not in estado.insights:  # una redacción por mes y sesión: las flechas no gastan
        estado.insights[mes] = redactar_insights(hechos_del_mes(df, mes), estado.llm)
    for texto in estado.insights[mes]:
        st.markdown(f"💡 {texto}")


def _linea_capas(resumen: ResumenCategorizacion) -> None:
    capas = resumen.por_capa
    partes = [
        f"{capas.get('reglas', 0)} por reglas",
        f"{capas.get('cache', 0)} por caché",
        f"{capas.get('llm', 0)} por IA",
    ]
    if capas.get("sin_llm"):
        partes.append(f"{capas['sin_llm']} sin IA disponible")
    costo = f"US$ {resumen.costo_usd:.4f}" if resumen.costo_usd is not None else "costo n/d"
    st.caption(f"🤖 Categorización: {' · '.join(partes)} · {costo}")


# --- Pestañas ---


def _tab_gastos(df: pd.DataFrame, mes: str) -> None:
    categorias, comercios = st.columns(2)
    with categorias:
        st.subheader("Gasto por categoría")
        _barras_horizontales(gasto_por_categoria(df, mes), "Categoría")
    with comercios:
        st.subheader("Top 5 comercios")
        _barras_horizontales(top_comercios(df, mes), "Comercio")


def _barras_horizontales(serie: pd.Series, etiqueta: str) -> None:
    if serie.empty:
        st.write("Sin gastos este mes.")
        return
    datos = serie.rename("gasto").rename_axis(etiqueta).reset_index()
    figura = px.bar(
        datos.iloc[::-1],  # Plotly dibuja de abajo hacia arriba: el mayor queda arriba
        x="gasto",
        y=etiqueta,
        orientation="h",
        text=datos.iloc[::-1]["gasto"].map(formatear_soles),
        color_discrete_sequence=[COLOR_UNICO],
    )
    figura.update_traces(
        textposition="outside", cliponaxis=False, hovertemplate="%{y}: S/ %{x:,.2f}<extra></extra>"
    )
    figura.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        xaxis_showticklabels=False,
        margin={"l": 0, "r": 40, "t": 10, "b": 0},
        bargap=0.35,
    )
    st.plotly_chart(figura, width="stretch")


def _tab_evolucion(df: pd.DataFrame) -> None:
    st.subheader("Evolución mensual")
    evolucion = evolucion_mensual(df).reset_index()
    evolucion["mes"] = evolucion["mes"].map(lambda m: nombre_mes(m).capitalize())
    largo = evolucion.melt(id_vars="mes", var_name="serie", value_name="soles")
    largo["serie"] = largo["serie"].str.capitalize()
    figura = px.line(
        largo,
        x="mes",
        y="soles",
        color="serie",
        markers=True,
        color_discrete_map={k.capitalize(): v for k, v in COLORES.items()},
    )
    figura.update_traces(
        line_width=2,
        marker_size=8,
        hovertemplate="%{x}: S/ %{y:,.2f}<extra>%{fullData.name}</extra>",
    )
    figura.update_layout(
        xaxis_title=None,
        yaxis_title="S/",
        legend_title_text=None,
        hovermode="x unified",
        margin={"t": 10},
    )
    st.plotly_chart(figura, width="stretch")
    st.dataframe(evolucion_mensual(df).map(formatear_soles), width="stretch")


def _tab_suscripciones(df: pd.DataFrame) -> None:
    st.subheader("Suscripciones y cobros recurrentes")
    suscripciones = detectar_suscripciones(df)
    if suscripciones.empty:
        st.write("No detectamos cobros recurrentes.")
        return
    total = suscripciones["costo_anual"].sum()
    st.metric("Costo anual estimado", formatear_soles(total))
    tabla = suscripciones.assign(
        monto_mensual=suscripciones["monto_mensual"].map(formatear_soles),
        costo_anual=suscripciones["costo_anual"].map(formatear_soles),
        meses=suscripciones["meses"].map(lambda ms: ", ".join(nombre_mes(m) for m in ms)),
    )
    st.dataframe(
        tabla,
        hide_index=True,
        width="stretch",
        column_order=[
            "descripcion",
            "monto_mensual",
            "costo_anual",
            "origen",
            "categoria",
            "meses",
        ],
        column_config={
            "descripcion": "Comercio",
            "categoria": "Categoría",
            "monto_mensual": "Mensual",
            "meses": "Meses",
            "costo_anual": "Anual (estimado)",
            "origen": "Detectada por",
        },
    )
    st.caption(
        "«patrón»: se cobró un monto similar (±10%) en meses consecutivos. "
        "«categoría»: está categorizada como suscripción, pero todavía no vemos que se repita."
    )


def _tab_ia(resumen: ResumenCategorizacion) -> None:
    st.subheader("Cómo se categorizaron tus transacciones")
    capas = resumen.por_capa
    reglas, cache, llm = st.columns(3)
    reglas.metric("Reglas (gratis)", capas.get("reglas", 0))
    cache.metric("Caché (gratis)", capas.get("cache", 0))
    llm.metric("IA", capas.get("llm", 0) + capas.get("sin_llm", 0))

    uso = st.session_state.llm.uso
    costo_sesion = uso.costo_usd(st.session_state.llm.modelo)
    st.write(
        f"- Descripciones únicas enviadas al modelo: **{resumen.descripciones_enviadas}**\n"
        f"- Tokens de la categorización: {resumen.tokens_entrada:,} de entrada · "
        f"{resumen.tokens_salida:,} de salida\n"
        f"- Transacciones con baja confianza (van a «Otros»): {resumen.baja_confianza}\n"
        f"- Costo total de IA en esta sesión (incluye la redacción de insights): "
        + (f"US$ {costo_sesion:.4f}" if costo_sesion is not None else "n/d")
    )
    st.info(
        "🔒 Al modelo solo se envía la descripción de cada comercio desconocido y si es cargo "
        "o abono. Nunca montos, fechas ni transferencias a personas."
    )


def _tab_transacciones(df: pd.DataFrame, mes: str) -> None:
    del_mes = df[df["mes"] == mes].sort_values("fecha")
    st.dataframe(
        del_mes[["fecha", "descripcion_original", "monto_pen", "categoria", "capa", "confianza"]],
        hide_index=True,
        width="stretch",
        column_config={
            "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
            "descripcion_original": "Descripción",
            "monto_pen": st.column_config.NumberColumn("Monto (S/)", format="%.2f"),
            "categoria": "Categoría",
            "capa": "Resuelto por",
            "confianza": st.column_config.NumberColumn("Confianza", format="%.2f"),
        },
    )


main()
