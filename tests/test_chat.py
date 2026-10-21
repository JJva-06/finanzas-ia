import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.chat import (
    AVISO_PARCIAL,
    MAX_HERRAMIENTAS,
    HerramientaError,
    buscar_transacciones,
    definiciones_herramientas,
    ejecutar_herramienta,
    gasto_por_categoria_tool,
    listar_suscripciones,
    listar_transacciones,
    responder,
    resumen_mensual,
    top_gastos,
)
from src.llm import LLMNoDisponible, Uso


def _df() -> pd.DataFrame:
    filas = [
        ("05/08/2026", "DLC NETFLIX.COM", -44.9, "Suscripciones"),
        ("10/08/2026", "RAPPI PERU SAC", -30.0, "Comida y delivery"),
        ("05/09/2026", "DLC NETFLIX.COM", -44.9, "Suscripciones"),
        ("12/09/2026", "RAPPI PERU SAC", -50.0, "Comida y delivery"),
        ("20/09/2026", "RAPPI PERU SAC", -20.0, "Comida y delivery"),
        ("25/09/2026", "YAPE-ANA TORRES", -100.0, "Transferencias"),
        ("30/09/2026", "ABONO PLANILLA", 1500.0, "Ingresos"),
    ]
    df = pd.DataFrame(filas, columns=["fecha", "descripcion", "monto_pen", "categoria"])
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")
    df["mes"] = df["fecha"].dt.strftime("%Y-%m")
    df["id"], df["capa"], df["confianza"] = range(len(df)), "reglas", 1.0
    return df


DF = _df()


# --- Herramientas ---


def test_gasto_por_categoria_en_un_mes_y_en_todo_el_periodo() -> None:
    sept = gasto_por_categoria_tool(DF, "Comida y delivery", "2026-09")
    assert sept == {
        "categoria": "Comida y delivery",
        "periodo": "septiembre 2026",
        "total": "S/ 70.00",
        "cantidad": 2,
    }
    assert gasto_por_categoria_tool(DF, "Comida y delivery")["total"] == "S/ 100.00"


def test_top_gastos_devuelve_filas_ordenadas_y_limitadas() -> None:
    resultado = top_gastos(DF, n=50, mes="2026-09")
    montos = [g["monto"] for g in resultado["gastos"]]
    assert montos == ["-S/ 100.00", "-S/ 50.00", "-S/ 44.90", "-S/ 20.00"]
    assert len(top_gastos(DF, n=2)["gastos"]) == 2


def test_resumen_mensual() -> None:
    resumen = resumen_mensual(DF, "2026-09")
    assert resumen["gasto"] == "S/ 214.90"
    assert resumen["ingreso"] == "S/ 1,500.00"
    assert resumen["balance"] == "S/ 1,285.10"
    assert resumen["variacion_gasto_vs_mes_anterior"] == "+186.9%"
    assert list(resumen["categorias_principales"])[0] == "Transferencias"


def test_listar_suscripciones() -> None:
    resultado = listar_suscripciones(DF)
    assert [s["comercio"] for s in resultado["suscripciones"]] == ["DLC NETFLIX.COM"]
    assert resultado["costo_anual_total"] == "S/ 538.80"


def test_buscar_transacciones_devuelve_resumen_sin_lista() -> None:
    resultado = buscar_transacciones(DF, "rappi")
    assert resultado["coincidencias"] == 3
    assert resultado["total_cargos"] == "S/ 100.00"
    assert resultado["promedio_cargo"] == "S/ 33.33"
    assert resultado["cargo_maximo"] == {
        "fecha": "12/09/2026",
        "comercio": "RAPPI PERU SAC",
        "monto": "-S/ 50.00",
    }
    assert resultado["cargo_minimo"]["fecha"] == "20/09/2026"
    assert resultado["mas_reciente"]["fecha"] == "20/09/2026"
    assert "transacciones" not in resultado


def test_listar_transacciones_con_tope_y_nota() -> None:
    resultado = listar_transacciones(DF, "rappi", n=2)
    assert [t["fecha"] for t in resultado["transacciones"]] == ["20/09/2026", "12/09/2026"]
    assert resultado["nota"] == "se muestran 2 de 3, las más recientes"


def test_ninguna_herramienta_devuelve_columnas_internas() -> None:
    salidas = json.dumps(
        [
            top_gastos(DF, 10),
            buscar_transacciones(DF, "rappi"),
            listar_transacciones(DF, "rappi"),
            listar_suscripciones(DF),
        ]
    )
    for interna in ("capa", "confianza", '"id"', "descripcion_original"):
        assert interna not in salidas


@pytest.mark.parametrize(
    ("nombre", "argumentos", "mensaje"),
    [
        ("resumen_mensual", {"mes": "2025-01"}, "No hay datos"),
        ("gasto_por_categoria", {"categoria": "Restaurantes"}, "no existe"),
        ("buscar_transacciones", {"texto": "spotify"}, "No hay transacciones"),
        ("borrar_todo", {}, "no existe"),
        ("top_gastos", {"cantidad": 3}, "Argumentos inválidos"),
    ],
)
def test_errores_de_herramientas(nombre: str, argumentos: dict, mensaje: str) -> None:
    with pytest.raises(HerramientaError, match=mensaje):
        ejecutar_herramienta(DF, nombre, argumentos)


def test_definiciones_usan_los_meses_y_categorias_validos() -> None:
    definiciones = {d["name"]: d for d in definiciones_herramientas(DF)}
    assert len(definiciones) == 6
    mes = definiciones["resumen_mensual"]["input_schema"]["properties"]["mes"]
    assert mes["enum"] == ["2026-08", "2026-09"]
    categoria = definiciones["gasto_por_categoria"]["input_schema"]["properties"]["categoria"]
    assert "Comida y delivery" in categoria["enum"]


# --- Bucle de conversación (con un doble: nunca se llama a la API real) ---


def _texto(texto: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=texto)], stop_reason="end_turn"
    )


def _pide(*pedidos: tuple[str, dict]) -> SimpleNamespace:
    bloques = [
        SimpleNamespace(type="tool_use", id=f"t{i}", name=nombre, input=argumentos)
        for i, (nombre, argumentos) in enumerate(pedidos)
    ]
    return SimpleNamespace(content=bloques, stop_reason="tool_use")


class LLMChatFalso:
    def __init__(self, *respuestas: object) -> None:
        self.modelo = "claude-haiku-5-5"
        self.uso = Uso()
        self.respuestas = list(respuestas)
        self.llamadas: list[dict] = []

    def conversar(self, system, messages, tools, tool_choice=None):
        self.llamadas.append({"messages": list(messages), "tool_choice": tool_choice})
        respuesta = self.respuestas.pop(0)
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta


def _resultados_enviados(llm: LLMChatFalso, llamada: int) -> list[dict]:
    return llm.llamadas[llamada]["messages"][-1]["content"]


def test_una_herramienta_y_respuesta() -> None:
    llm = LLMChatFalso(
        _pide(("gasto_por_categoria", {"categoria": "Comida y delivery", "mes": "2026-09"})),
        _texto("En septiembre gastaste S/ 70.00 en comida y delivery."),
    )
    historial: list[dict] = []

    respuesta = responder("¿Cuánto gasté en delivery en septiembre?", historial, DF, llm)

    assert respuesta.texto == "En septiembre gastaste S/ 70.00 en comida y delivery."
    assert respuesta.herramientas == ["gasto_por_categoria"]
    resultado = _resultados_enviados(llm, 1)[0]
    assert resultado["tool_use_id"] == "t0"
    assert json.loads(resultado["content"])["total"] == "S/ 70.00"
    assert [m["role"] for m in historial] == ["user", "assistant"]


def test_varias_herramientas_en_una_vuelta_van_en_un_solo_mensaje() -> None:
    llm = LLMChatFalso(
        _pide(("resumen_mensual", {"mes": "2026-09"}), ("listar_suscripciones", {})),
        _texto("Listo."),
    )
    respuesta = responder("Resumen y suscripciones", [], DF, llm)
    assert respuesta.herramientas == ["resumen_mensual", "listar_suscripciones"]
    assert len(_resultados_enviados(llm, 1)) == 2


def test_error_de_herramienta_se_devuelve_al_modelo() -> None:
    llm = LLMChatFalso(
        _pide(("resumen_mensual", {"mes": "2025-01"})),
        _texto("No tengo datos de enero de 2025."),
    )
    respuesta = responder("¿Y enero 2025?", [], DF, llm)
    resultado = _resultados_enviados(llm, 1)[0]
    assert resultado["is_error"] is True
    assert "No hay datos" in resultado["content"]
    assert respuesta.texto == "No tengo datos de enero de 2025."


def test_limite_de_herramientas_fuerza_respuesta_y_avisa() -> None:
    pedidos = [_pide(("listar_suscripciones", {})) for _ in range(MAX_HERRAMIENTAS)]
    llm = LLMChatFalso(*pedidos, _texto("Esto es lo que encontré."))

    respuesta = responder("Pregunta enorme", [], DF, llm)

    assert len(respuesta.herramientas) == MAX_HERRAMIENTAS
    assert respuesta.parcial
    assert AVISO_PARCIAL in respuesta.texto
    assert llm.llamadas[-1]["tool_choice"] == {"type": "none"}
    assert all(c["tool_choice"] is None for c in llm.llamadas[:-1])


def test_pedidos_extra_en_la_misma_vuelta_no_se_ejecutan() -> None:
    pedidos = [("listar_suscripciones", {})] * (MAX_HERRAMIENTAS + 2)
    llm = LLMChatFalso(_pide(*pedidos), _texto("Ok."))
    respuesta = responder("muchas", [], DF, llm)
    resultados = _resultados_enviados(llm, 1)
    assert len(respuesta.herramientas) == MAX_HERRAMIENTAS
    assert sum(r.get("is_error", False) for r in resultados) == 2


def test_historial_solo_guarda_texto_y_se_recorta() -> None:
    historial = [{"role": r, "content": "x"} for _ in range(30) for r in ("user", "assistant")]
    llm = LLMChatFalso(_texto("Hola."))
    responder("¿Qué tal?", historial, DF, llm)
    enviados = llm.llamadas[0]["messages"]
    assert len(enviados) == 21  # 10 intercambios + la pregunta nueva
    assert enviados[0]["role"] == "user"
    assert historial[-1] == {"role": "assistant", "content": "Hola."}


def test_api_caida_se_propaga_para_la_ui() -> None:
    llm = LLMChatFalso(LLMNoDisponible("No se pudo conectar con el servicio de IA."))
    historial: list[dict] = []
    with pytest.raises(LLMNoDisponible):
        responder("Hola", historial, DF, llm)
    assert historial == []
