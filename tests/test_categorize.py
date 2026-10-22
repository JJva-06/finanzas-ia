import json

import pandas as pd
import pytest

from scripts.generate_sample import generar
from src.categorize import (
    MAX_DESCRIPCIONES_LLM,
    REGLAS,
    CacheCategorias,
    aplicar_reglas,
    aplicar_resultados_llm,
    categorizar,
    categorizar_con_llm,
    categorizar_por_cache,
    categorizar_por_reglas,
    pendientes_unicos,
    tokenizar,
)
from src.ingest import DESCRIPCION_VACIA, cargar_csv, limpiar_descripcion
from src.llm import LLMNoDisponible, Uso
from src.schemas import Categoria, RespuestaLote, ResultadoItem


def test_tokenizar_separa_por_signos() -> None:
    assert tokenizar("DLC NETFLIX.COM") == ["DLC", "NETFLIX", "COM"]
    assert tokenizar("YAPE-ANA TORRES") == ["YAPE", "ANA", "TORRES"]
    assert tokenizar("TAMBO+ LARCO") == ["TAMBO", "LARCO"]


@pytest.mark.parametrize(
    ("descripcion", "esperada"),
    [
        ("RAPPI PERU SAC", Categoria.COMIDA),
        ("RAPPI RAPPI PRO", Categoria.SUSCRIPCIONES),
        ("UBER TRIP", Categoria.TRANSPORTE),
        ("UBER EATS PEDIDO", Categoria.COMIDA),
        ("DLC NETFLIX.COM", Categoria.SUSCRIPCIONES),
        ("YAPE-ANA TORRES", Categoria.TRANSFERENCIAS),
        ("TRANSFERENCIA RECIBIDA M QUISPE", Categoria.TRANSFERENCIAS),
        ("ABONO PLANILLA INNOVA DATA SAC", Categoria.INGRESOS),
        ("POS PLAZA VEA SURCO", Categoria.SUPERMERCADO),
        ("MOVISTAR MOVIL PLAN POSTPAGO", Categoria.SERVICIOS),
        ("TARJETA METROPOLITANO", Categoria.TRANSPORTE),
    ],
)
def test_aplicar_reglas(descripcion: str, esperada: Categoria) -> None:
    assert aplicar_reglas(descripcion) == esperada


def test_solo_palabras_completas() -> None:
    assert aplicar_reglas("LIBRERIA ACLARO") is None


def test_frase_requiere_palabras_consecutivas() -> None:
    assert aplicar_reglas("PLAZA SAN MIGUEL VEA") is None


def test_sin_descripcion_va_a_otros() -> None:
    assert aplicar_reglas(DESCRIPCION_VACIA) == Categoria.OTROS


def test_comercio_desconocido_queda_pendiente() -> None:
    assert aplicar_reglas("IZI CHIFA DRAGON DORADO") is None


def test_reglas_sin_frases_repetidas() -> None:
    frases = [tuple(tokenizar(frase)) for frase, _ in REGLAS]
    assert len(frases) == len(set(frases))


def test_categorizar_por_reglas_agrega_columnas() -> None:
    df = pd.DataFrame({"descripcion": ["RAPPI PERU SAC", "BODEGA DON LUCHO"]})
    resultado = categorizar_por_reglas(df)

    assert resultado.loc[0, "categoria"] == "Comida y delivery"
    assert resultado.loc[0, "capa"] == "reglas"
    assert resultado.loc[0, "confianza"] == 1.0
    assert resultado.loc[1, ["categoria", "capa"]].isna().all()
    assert pd.isna(resultado.loc[1, "confianza"])
    assert "categoria" not in df.columns  # no modifica el original


def test_reglas_resuelven_exactamente_los_comercios_conocidos_del_ejemplo() -> None:
    """Las reglas cubren los comercios conocidos del generador y dejan los locales al LLM."""
    df, _ = generar()
    resueltas = df["descripcion"].map(limpiar_descripcion).map(aplicar_reglas).notna()
    pd.testing.assert_series_equal(resueltas, df["conocido"], check_names=False)


def _pendientes(*descripciones: str) -> pd.DataFrame:
    return categorizar_por_reglas(pd.DataFrame({"descripcion": list(descripciones)}))


def test_pendientes_unicos_deduplica_y_omite_resueltas() -> None:
    df = _pendientes("BODEGA DON LUCHO", "RAPPI PERU SAC", "BODEGA DON LUCHO", "IZI CHIFA")
    assert pendientes_unicos(df) == ["BODEGA DON LUCHO", "IZI CHIFA"]


def test_resultados_llm_primera_vez_llm_y_repeticiones_cache() -> None:
    df = _pendientes("IZI CHIFA", "RAPPI PERU SAC", "IZI CHIFA", "IZI CHIFA")
    cache = CacheCategorias()

    resultado = aplicar_resultados_llm(df, {"IZI CHIFA": (Categoria.COMIDA, 0.9)}, cache)

    assert list(resultado["capa"]) == ["llm", "reglas", "cache", "cache"]
    assert set(resultado["categoria"]) == {"Comida y delivery"}
    assert cache.buscar("IZI CHIFA") == (Categoria.COMIDA, 0.9)


@pytest.mark.parametrize(
    ("confianza", "categoria"),
    [(0.5, Categoria.OTROS), (0.51, Categoria.SUPERMERCADO), (0.2, Categoria.OTROS)],
)
def test_umbral_de_confianza(confianza: float, categoria: Categoria) -> None:
    df = _pendientes("BODEGA DON LUCHO")
    cache = CacheCategorias()

    resultado = aplicar_resultados_llm(
        df, {"BODEGA DON LUCHO": (Categoria.SUPERMERCADO, confianza)}, cache
    )

    assert resultado.loc[0, "categoria"] == categoria
    assert resultado.loc[0, "confianza"] == confianza
    assert resultado.loc[0, "capa"] == "llm"
    # Lo dudoso se cachea como Otros: nunca se guarda la categoría adivinada.
    assert cache.buscar("BODEGA DON LUCHO") == (categoria, confianza)


def test_descripcion_sin_respuesta_queda_pendiente() -> None:
    resultado = aplicar_resultados_llm(_pendientes("IZI CHIFA"), {}, CacheCategorias())
    assert pd.isna(resultado.loc[0, "categoria"])


def test_segunda_corrida_resuelve_todo_con_cache() -> None:
    cache = CacheCategorias()
    primera = _pendientes("IZI CHIFA", "BODEGA DON LUCHO")
    aplicar_resultados_llm(primera, {"IZI CHIFA": (Categoria.COMIDA, 0.9)}, cache)

    segunda = categorizar_por_cache(_pendientes("IZI CHIFA", "BODEGA DON LUCHO"), cache)

    assert segunda.loc[0, "capa"] == "cache"
    assert segunda.loc[0, "categoria"] == "Comida y delivery"
    assert pd.isna(segunda.loc[1, "categoria"])  # no estaba en la caché
    assert pendientes_unicos(segunda) == ["BODEGA DON LUCHO"]


def test_desglose_por_capa_en_el_archivo_de_ejemplo() -> None:
    df = categorizar_por_reglas(cargar_csv("data/sample/estado_cuenta_ejemplo.csv").df)
    cache = CacheCategorias()
    df = categorizar_por_cache(df, cache)
    unicos = pendientes_unicos(df)

    resultado = aplicar_resultados_llm(df, {d: (Categoria.OTROS, 0.9) for d in unicos}, cache)

    capas = resultado["capa"].value_counts()
    assert len(unicos) == capas["llm"] == 8
    assert capas["reglas"] + capas["cache"] + capas["llm"] == len(resultado)
    assert len(cache) == 8


# --- Capa LLM (con un doble: nunca se llama a la API real) ---


class LLMFalso:
    """Doble de ClienteLLM: cada llamada consume la siguiente acción programada."""

    def __init__(self, *acciones: object) -> None:
        self.modelo = "claude-haiku-5-5"
        self.uso = Uso()
        self.acciones = list(acciones)
        self.lotes: list[list[dict]] = []

    def parse(self, system: str, contenido: str, formato: type) -> RespuestaLote | None:
        lote = json.loads(contenido)
        self.lotes.append(lote)
        self.uso.llamadas += 1
        self.uso.tokens_entrada += 100
        self.uso.tokens_salida += 20
        accion = self.acciones.pop(0) if self.acciones else responder_todo()
        if isinstance(accion, Exception):
            raise accion
        return accion(lote)


def responder_todo(categoria: Categoria = Categoria.COMIDA, confianza: float = 0.9):
    def accion(lote: list[dict]) -> RespuestaLote:
        return RespuestaLote(
            resultados=[
                ResultadoItem(id=item["id"], categoria=categoria, confianza=confianza)
                for item in lote
            ]
        )

    return accion


def omitir(*descripciones: str):
    def accion(lote: list[dict]) -> RespuestaLote:
        completa = responder_todo()(lote)
        quedan = [r for r in completa.resultados if lote[r.id]["descripcion"] not in descripciones]
        return RespuestaLote(resultados=quedan)

    return accion


def test_llm_recibe_solo_descripcion_y_tipo() -> None:
    llm = LLMFalso()
    categorizar_con_llm([("IZI CHIFA", "cargo"), ("ABONO SERV XYZ", "abono")], llm)

    assert llm.lotes == [
        [
            {"id": 0, "descripcion": "IZI CHIFA", "tipo": "cargo"},
            {"id": 1, "descripcion": "ABONO SERV XYZ", "tipo": "abono"},
        ]
    ]


def test_reintenta_solo_lo_que_falto() -> None:
    llm = LLMFalso(omitir("B"), responder_todo(Categoria.SALUD))

    resultados = categorizar_con_llm([("A", "cargo"), ("B", "cargo"), ("C", "cargo")], llm)

    assert [item["descripcion"] for item in llm.lotes[1]] == ["B"]
    assert resultados["A"] == (Categoria.COMIDA, 0.9)
    assert resultados["B"] == (Categoria.SALUD, 0.9)


def test_lo_que_falla_dos_veces_va_a_otros() -> None:
    llm = LLMFalso(omitir("B"), omitir("B"))
    resultados = categorizar_con_llm([("A", "cargo"), ("B", "cargo")], llm)
    assert resultados["B"] == (Categoria.OTROS, 0.0)
    assert len(llm.lotes) == 2


def test_salida_invalida_reintenta_el_lote_completo() -> None:
    llm = LLMFalso(lambda lote: None, responder_todo())
    resultados = categorizar_con_llm([("A", "cargo"), ("B", "cargo")], llm)
    assert len(llm.lotes[1]) == 2
    assert resultados["A"][0] == Categoria.COMIDA


@pytest.mark.parametrize(
    "invalidos",
    [
        [ResultadoItem(id=5, categoria=Categoria.COMIDA, confianza=0.9)],
        [
            ResultadoItem(id=0, categoria=Categoria.COMIDA, confianza=0.9),
            ResultadoItem(id=0, categoria=Categoria.SALUD, confianza=0.9),
        ],
        [ResultadoItem(id=0, categoria=Categoria.COMIDA, confianza=1.7)],
    ],
)
def test_ids_ajenos_repetidos_o_confianza_fuera_de_rango_no_valen(
    invalidos: list[ResultadoItem],
) -> None:
    llm = LLMFalso(lambda lote: RespuestaLote(resultados=invalidos), omitir("A"))
    assert categorizar_con_llm([("A", "cargo")], llm)["A"] == (Categoria.OTROS, 0.0)


def test_lotes_de_40() -> None:
    llm = LLMFalso()
    categorizar_con_llm([(f"COMERCIO {i}", "cargo") for i in range(45)], llm)
    assert [len(lote) for lote in llm.lotes] == [40, 5]


def test_tipo_sale_del_signo() -> None:
    df = pd.DataFrame({"descripcion": ["X", "Y", "Z", "Z"], "monto": [-10.0, 50.0, -5.0, 5.0]})
    llm = LLMFalso()
    categorizar(df, CacheCategorias(), llm)
    tipos = {item["descripcion"]: item["tipo"] for item in llm.lotes[0]}
    assert tipos == {"X": "cargo", "Y": "abono", "Z": "mixto"}


def test_categorizar_archivo_de_ejemplo() -> None:
    df = cargar_csv("data/sample/estado_cuenta_ejemplo.csv").df

    resultado, resumen = categorizar(df, CacheCategorias(), LLMFalso())

    assert resultado["categoria"].notna().all()
    assert resumen.por_capa == {"reglas": 124, "cache": 35, "llm": 8}
    assert resumen.descripciones_enviadas == 8
    assert (resumen.tokens_entrada, resumen.tokens_salida) == (100, 20)
    assert resumen.costo_usd == pytest.approx((100 * 0.10 + 20 * 0.50) / 1_000_000)
    assert resumen.avisos == []


def test_segunda_corrida_no_llama_al_llm() -> None:
    df = cargar_csv("data/sample/estado_cuenta_ejemplo.csv").df
    cache, llm = CacheCategorias(), LLMFalso()
    categorizar(df, cache, llm)

    _, resumen = categorizar(df, cache, llm)

    assert resumen.descripciones_enviadas == 0
    assert resumen.tokens_entrada == 0
    assert len(llm.lotes) == 1


def test_baja_confianza_se_cuenta() -> None:
    df = pd.DataFrame({"descripcion": ["X", "X"], "monto": [-1.0, -2.0]})
    llm = LLMFalso(responder_todo(confianza=0.3))
    resultado, resumen = categorizar(df, CacheCategorias(), llm)
    assert set(resultado["categoria"]) == {"Otros"}
    assert resumen.baja_confianza == 2  # la fila del LLM y su repetición desde la caché


def test_lo_dudoso_no_se_vuelve_a_consultar_en_la_sesion() -> None:
    df = pd.DataFrame({"descripcion": ["NIUBIZ BARBERIA"], "monto": [-25.0]})
    cache, llm = CacheCategorias(), LLMFalso(responder_todo(Categoria.SALUD, 0.4))
    categorizar(df, cache, llm)

    resultado, resumen = categorizar(df, cache, llm)

    assert len(llm.lotes) == 1
    assert resumen.descripciones_enviadas == 0
    assert resultado.loc[0, "categoria"] == "Otros"
    assert resultado.loc[0, "capa"] == "cache"
    assert resumen.baja_confianza == 1


@pytest.mark.parametrize("llm", [None, LLMFalso(LLMNoDisponible("Sin conexión."))])
def test_sin_llm_la_app_sigue_funcionando(llm: LLMFalso | None) -> None:
    df = cargar_csv("data/sample/estado_cuenta_ejemplo.csv").df

    resultado, resumen = categorizar(df, CacheCategorias(), llm)

    assert resultado["categoria"].notna().all()
    assert resumen.por_capa == {"reglas": 124, "sin_llm": 43}
    assert "8 comercios desconocidos quedaron en 'Otros'" in resumen.avisos[0]


def test_al_llm_van_primero_las_mas_frecuentes() -> None:
    df = pd.DataFrame({"descripcion": ["RARO", "COMUN", "COMUN", "COMUN"], "monto": [-1.0] * 4})
    llm = LLMFalso()
    categorizar(df, CacheCategorias(), llm)
    assert [item["descripcion"] for item in llm.lotes[0]] == ["COMUN", "RARO"]


def test_tope_de_descripciones_al_llm() -> None:
    extra = 7
    descripciones = [f"COMERCIO {i}" for i in range(MAX_DESCRIPCIONES_LLM + extra)]
    # El primero se repite: tiene que estar entre los que sí van al LLM.
    df = pd.DataFrame(
        {"descripcion": descripciones + ["COMERCIO 0"], "monto": [-1.0] * (len(descripciones) + 1)}
    )
    llm = LLMFalso()

    resultado, resumen = categorizar(df, CacheCategorias(), llm)

    enviadas = [item["descripcion"] for lote in llm.lotes for item in lote]
    assert len(enviadas) == resumen.descripciones_enviadas == MAX_DESCRIPCIONES_LLM
    assert enviadas[0] == "COMERCIO 0"
    assert resumen.por_capa["sin_llm"] == extra
    assert resultado["categoria"].notna().all()
    assert any(f"{extra} comercios poco frecuentes" in aviso for aviso in resumen.avisos)
