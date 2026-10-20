import pandas as pd
import pytest

from scripts.generate_sample import generar
from src.categorize import (
    REGLAS,
    CacheCategorias,
    aplicar_reglas,
    aplicar_resultados_llm,
    categorizar_por_cache,
    categorizar_por_reglas,
    pendientes_unicos,
    tokenizar,
)
from src.ingest import DESCRIPCION_VACIA, cargar_csv, limpiar_descripcion
from src.schemas import Categoria


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
    ("confianza", "categoria", "en_cache"),
    [(0.5, "Otros", False), (0.51, "Supermercado", True), (0.2, "Otros", False)],
)
def test_umbral_de_confianza(confianza: float, categoria: str, en_cache: bool) -> None:
    df = _pendientes("BODEGA DON LUCHO")
    cache = CacheCategorias()

    resultado = aplicar_resultados_llm(
        df, {"BODEGA DON LUCHO": (Categoria.SUPERMERCADO, confianza)}, cache
    )

    assert resultado.loc[0, "categoria"] == categoria
    assert resultado.loc[0, "confianza"] == confianza
    assert resultado.loc[0, "capa"] == "llm"
    assert (cache.buscar("BODEGA DON LUCHO") is not None) == en_cache


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
