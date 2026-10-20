import pandas as pd
import pytest

from scripts.generate_sample import generar
from src.categorize import REGLAS, aplicar_reglas, categorizar_por_reglas, tokenizar
from src.ingest import DESCRIPCION_VACIA, limpiar_descripcion
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
