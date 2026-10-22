"""Entrada de Streamlit: subir un estado de cuenta (o usar el de ejemplo) → dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src import tema
from src.categorize import CacheCategorias, ResumenCategorizacion, categorizar
from src.chat import responder
from src.ingest import IngestError, IngestResult, cargar_csv, combinar_archivos
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
from src.llm import ClienteLLM, LLMNoDisponible
from src.narrativa import redactar_insights

ARCHIVO_EJEMPLO = Path("data/sample/estado_cuenta_ejemplo.csv")
PREGUNTAS_EJEMPLO = (
    "¿Cuánto gasté en delivery en septiembre?",
    "¿Cuál fue mi gasto más grande?",
    "¿Qué suscripciones tengo?",
)

st.set_page_config(page_title="Mis finanzas · boleta con IA", page_icon="🧾", layout="centered")


def main() -> None:
    _inicializar_sesion()
    st.markdown(tema.hoja_de_estilos(), unsafe_allow_html=True)
    _html(tema.cabecera())
    _enlace_modo_tecnico()
    _html(tema.corte(doble=True))
    _seccion_carga()

    datos = st.session_state.datos
    if datos is None:
        _html(
            '<p class="aviso-pequeno">Sube un CSV con columnas fecha, descripcion, monto '
            "o usa los datos de ejemplo (100% sintéticos).</p>"
        )
        return

    df: pd.DataFrame = datos["df"]
    _mostrar_avisos(datos["avisos"], df)
    _html(tema.corte())
    mes = _selector_mes(df)
    _kpis(df, mes)
    _insights(df, mes)
    _html(tema.corte(doble=True))

    nombres = ["Pregúntale a tus gastos", "Gastos", "Evolución", "Suscripciones", "Movimientos"]
    if _tecnico():
        nombres.append("Cómo categorizó la IA")
    pestanas = st.tabs(nombres)
    with pestanas[0]:
        _tab_chat(df)
    with pestanas[1]:
        _tab_gastos(df, mes)
    with pestanas[2]:
        _tab_evolucion(df)
    with pestanas[3]:
        _tab_suscripciones(df)
    with pestanas[4]:
        _tab_movimientos(df, mes)
    if _tecnico():
        with pestanas[5]:
            _tab_ia(datos["resumen"])

    _html(tema.corte(doble=True))
    _pie(datos["resumen"])


def _html(contenido: str) -> None:
    st.markdown(contenido, unsafe_allow_html=True)


# --- Modo técnico: la vista del usuario es limpia; las métricas internas, a un clic ---


def _tecnico() -> bool:
    return st.session_state.tecnico


def _enlace_modo_tecnico() -> None:
    # Botón con aspecto de enlace (no un link real): cambia de vista sin recargar ni
    # perder la sesión, que tiene los datos ya categorizados.
    texto = "‹ volver a la vista simple" if _tecnico() else "ver cómo funciona por dentro ›"
    with st.container(key="modo_tecnico"):
        st.button(texto, type="tertiary", on_click=_alternar_modo_tecnico)


def _alternar_modo_tecnico() -> None:
    st.session_state.tecnico = not st.session_state.tecnico


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
    estado.setdefault("chat_historial", [])  # lo que se manda al LLM (solo texto)
    estado.setdefault("chat_mensajes", [])  # lo que se muestra, con metadatos
    # ?tecnico=1 en la URL abre directo la vista técnica (link del README).
    estado.setdefault("tecnico", st.query_params.get("tecnico") == "1")


def _seccion_carga() -> None:
    subir, ejemplo = st.columns([3, 2], vertical_alignment="bottom")
    archivos = subir.file_uploader(
        "Sube tus estados de cuenta (CSV)", type="csv", accept_multiple_files=True
    )
    usar_ejemplo = ejemplo.button("Usar datos de ejemplo", type="primary", width="stretch")

    if usar_ejemplo:
        _procesar([(ARCHIVO_EJEMPLO.name, ARCHIVO_EJEMPLO.read_bytes())], origen=("ejemplo",))
    elif archivos:
        origen = tuple(a.file_id for a in archivos)
        if st.session_state.origen != origen:
            _procesar([(a.name, a.getvalue()) for a in archivos], origen=origen)


def _procesar(archivos: list[tuple[str, bytes]], origen: tuple[str, ...]) -> None:
    estado = st.session_state
    estado.origen = origen
    leidos: list[tuple[str, IngestResult]] = []
    for nombre, contenido in archivos:
        try:
            leidos.append((nombre, cargar_csv(contenido)))
        except IngestError as error:
            st.error(f"No pudimos leer {nombre}: {error}")
    if not leidos:
        estado.datos = None
        return
    ingesta = combinar_archivos(leidos) if len(leidos) > 1 else leidos[0][1]

    with st.spinner("Imprimiendo tu boleta: reglas, caché e IA…"):
        df, resumen = categorizar(ingesta.df, estado.cache, estado.llm)
    df = agregar_monto_pen(df, tipo_cambio_usd())

    estado.datos = {"df": df, "resumen": resumen, "avisos": ingesta.avisos + resumen.avisos}
    estado.insights = {}
    estado.chat_historial, estado.chat_mensajes = [], []  # otros datos, otra conversación
    estado.mes_idx = len(meses_disponibles(df)) - 1  # al entrar, el mes más reciente


def _mostrar_avisos(avisos: list[str], df: pd.DataFrame) -> None:
    if avisos:
        with st.expander(f"{len(avisos)} aviso(s) sobre tus archivos"):
            for aviso in avisos:
                st.write(f"- {aviso}")
    if hubo_conversion(df):
        _html(
            f'<p class="aviso-pequeno">Montos en US$ convertidos a soles con un tipo de cambio '
            f"referencial de {tipo_cambio_usd():.2f}.</p>"
        )


# --- Resumen del mes ---


def _selector_mes(df: pd.DataFrame) -> str:
    meses = meses_disponibles(df)
    indice = min(st.session_state.mes_idx, len(meses) - 1)
    mes = meses[indice]

    # La key le da a esta fila una clase CSS propia: en celular no se apila (ver boleta.css).
    with st.container(key="nav_mes"):
        anterior, titulo, siguiente = st.columns([1, 4, 1], vertical_alignment="center")
        anterior.button("◀", on_click=_mover_mes, args=(-1,), disabled=indice == 0, width="stretch")
        siguiente.button(
            "▶", on_click=_mover_mes, args=(1,), disabled=indice == len(meses) - 1, width="stretch"
        )
    anio = pd.Period(mes, freq="M").year
    with titulo:
        _html(tema.titulo_mes(f"{nombre_mes(mes)} {anio}"))
    return mes


def _mover_mes(paso: int) -> None:
    st.session_state.mes_idx += paso


def _kpis(df: pd.DataFrame, mes: str) -> None:
    kpis = kpis_mes(df, mes)
    sello = None
    if kpis.variacion_gasto is not None:
        anterior = nombre_mes(str(pd.Period(mes, freq="M") - 1))
        sello = f"Gasto {kpis.variacion_gasto:+.1%} vs. {anterior}"
    filas = [
        ("Ingreso del mes", formatear_soles(kpis.ingreso)),
        ("Gasto del mes", formatear_soles(-kpis.gasto)),
    ]
    _html(tema.lineas_kpi(filas, ("Balance", formatear_soles(kpis.balance)), sello))


def _insights(df: pd.DataFrame, mes: str) -> None:
    estado = st.session_state
    if mes not in estado.insights:  # una redacción por mes y sesión: las flechas no gastan
        estado.insights[mes] = redactar_insights(hechos_del_mes(df, mes), estado.llm)
    _html(tema.corte() + tema.notas(estado.insights[mes]))


def _pie(resumen: ResumenCategorizacion) -> None:
    if not _tecnico():
        _html(tema.pie([]))
        return
    capas = resumen.por_capa
    partes = [
        f"<strong>{capas.get('reglas', 0)}</strong> reglas",
        f"<strong>{capas.get('cache', 0)}</strong> caché",
        f"<strong>{capas.get('llm', 0)}</strong> IA",
    ]
    if capas.get("sin_llm"):
        partes.append(f"<strong>{capas['sin_llm']}</strong> sin IA disponible")
    costo = f"US$ {resumen.costo_usd:.4f}" if resumen.costo_usd is not None else "n/d"
    _html(
        tema.pie(
            [
                f"Procesado por: {' · '.join(partes)}",
                f"Costo de la categorización: <strong>{costo}</strong>",
            ]
        )
    )


# --- Pestañas ---


def _tab_gastos(df: pd.DataFrame, mes: str) -> None:
    st.subheader("Gasto por categoría")
    _barras_horizontales(gasto_por_categoria(df, mes), "Categoría")
    st.subheader("Top 5 comercios")
    _barras_horizontales(top_comercios(df, mes), "Comercio")


def _barras_horizontales(serie: pd.Series, etiqueta: str) -> None:
    if serie.empty:
        st.write("Sin gastos este mes.")
        return
    datos = serie.rename("gasto").rename_axis(etiqueta).reset_index().iloc[::-1]
    figura = px.bar(  # Plotly dibuja de abajo hacia arriba: el mayor queda arriba
        datos,
        x="gasto",
        y=etiqueta,
        orientation="h",
        text=datos["gasto"].map(formatear_soles),
        color_discrete_sequence=[tema.TINTA],
    )
    figura.update_traces(
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        hovertemplate="%{y}: S/ %{x:,.2f}<extra></extra>",
    )
    figura.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        xaxis_showticklabels=False,
        xaxis_showgrid=False,
        margin={"l": 0, "r": 70, "t": 10, "b": 0},
        bargap=0.45,
        height=max(180, 46 * len(datos)),
    )
    st.plotly_chart(
        tema.estilizar(figura), width="stretch", theme=None, config={"displayModeBar": False}
    )


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
        line_dash="serie",
        markers=True,
        color_discrete_map={s: v["color"] for s, v in tema.SERIES.items()},
        line_dash_map={s: v["dash"] for s, v in tema.SERIES.items()},
    )
    figura.update_traces(
        line_width=2,
        marker_size=8,
        hovertemplate="%{x}: S/ %{y:,.2f}<extra>%{fullData.name}</extra>",
    )
    figura.update_layout(xaxis_title=None, yaxis_title="S/", hovermode="x unified")
    st.plotly_chart(
        tema.estilizar(figura), width="stretch", theme=None, config={"displayModeBar": False}
    )
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
        column_order=(
            ["descripcion", "monto_mensual", "costo_anual", "origen", "categoria", "meses"]
            if _tecnico()
            else ["descripcion", "monto_mensual", "costo_anual", "meses"]
        ),
        column_config={
            "descripcion": "Comercio",
            "categoria": "Categoría",
            "monto_mensual": "Mensual",
            "meses": "Meses",
            "costo_anual": "Anual (estimado)",
            "origen": "Detectada por",
        },
    )
    if _tecnico():
        st.caption(
            "«patrón»: se cobró un monto similar (±10%) en meses consecutivos. "
            "«categoría»: está categorizada como suscripción, pero todavía no vemos que se repita."
        )


def _tab_ia(resumen: ResumenCategorizacion) -> None:
    st.subheader("Cómo se categorizaron tus movimientos")
    capas = resumen.por_capa
    reglas, cache, llm = st.columns(3)
    reglas.metric("Reglas · gratis", capas.get("reglas", 0))
    cache.metric("Caché · gratis", capas.get("cache", 0))
    llm.metric("IA", capas.get("llm", 0) + capas.get("sin_llm", 0))

    uso = st.session_state.llm.uso
    costo_sesion = uso.costo_usd(st.session_state.llm.modelo)
    st.write(
        f"- Descripciones únicas enviadas al modelo: **{resumen.descripciones_enviadas}**\n"
        f"- Tokens de la categorización: {resumen.tokens_entrada:,} de entrada · "
        f"{resumen.tokens_salida:,} de salida\n"
        f"- Movimientos con baja confianza (van a «Otros»): {resumen.baja_confianza}\n"
        f"- Costo total de IA en esta sesión (incluye insights y chat): "
        + (f"US$ {costo_sesion:.4f}" if costo_sesion is not None else "n/d")
    )
    st.info(
        "Al modelo solo se envía la descripción de cada comercio desconocido y si es cargo "
        "o abono. Nunca montos, fechas ni transferencias a personas."
    )


def _tab_movimientos(df: pd.DataFrame, mes: str) -> None:
    del_mes = df[df["mes"] == mes].sort_values("fecha")
    columnas = ["fecha", "descripcion_original", "monto_pen", "categoria"]
    if _tecnico():
        columnas += ["capa", "confianza"]
    st.dataframe(
        del_mes[columnas],
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


def _tab_chat(df: pd.DataFrame) -> None:
    pregunta = None
    for columna, ejemplo in zip(st.columns(len(PREGUNTAS_EJEMPLO)), PREGUNTAS_EJEMPLO, strict=True):
        if columna.button(ejemplo, width="stretch"):
            pregunta = ejemplo

    for mensaje in st.session_state.chat_mensajes:
        _mostrar_mensaje(mensaje)

    # El campo va después de los mensajes; al responder se vuelve a dibujar todo (st.rerun)
    # para que la conversación quede arriba y el campo siempre abajo.
    pregunta = st.chat_input("Escribe tu pregunta sobre tus gastos") or pregunta
    st.caption("Tu archivo nunca se envía a la IA: solo ve el resultado de cada consulta.")
    if pregunta:
        _responder_pregunta(pregunta, df)
        st.rerun()


def _responder_pregunta(pregunta: str, df: pd.DataFrame) -> None:
    estado = st.session_state
    pregunta_msg = {"rol": "user", "texto": pregunta}
    _mostrar_mensaje(pregunta_msg)

    uso, costo_antes = estado.llm.uso, estado.llm.uso.costo_usd(estado.llm.modelo) or 0.0
    with st.spinner("Consultando tus datos…"):
        try:
            respuesta = responder(pregunta, estado.chat_historial, df, estado.llm)
            detalle = ", ".join(respuesta.herramientas) or "ninguna"
            costo = (uso.costo_usd(estado.llm.modelo) or 0.0) - costo_antes
            mensaje = {
                "rol": "assistant",
                "texto": respuesta.texto,
                "meta": f"Herramientas: {detalle} · US$ {costo:.4f}",
            }
        except LLMNoDisponible as error:
            mensaje = {
                "rol": "assistant",
                "texto": f"{error} El resto del dashboard sigue funcionando.",
            }
    estado.chat_mensajes.extend([pregunta_msg, mensaje])


def _mostrar_mensaje(mensaje: dict) -> None:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["texto"])  # sin unsafe_allow_html: el HTML se muestra como texto
        if "meta" in mensaje and _tecnico():
            st.caption(mensaje["meta"])


main()
