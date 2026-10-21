import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.evaluate import (
    ETIQUETADAS,
    PREGUNTAS,
    accuracy_por,
    cargar_etiquetadas,
    extraer_numeros,
    menciona_fecha,
    puntuar_categorizacion,
    respuesta_correcta,
)

# --- Las respuestas esperadas del chat se recalculan con pandas sobre el CSV crudo ---


def _ejemplo() -> pd.DataFrame:
    df = pd.read_csv("data/sample/estado_cuenta_ejemplo.csv")
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")
    df["mes"] = df["fecha"].dt.strftime("%Y-%m")
    return df


DF = _ejemplo()
CARGOS = DF[DF["monto"] < 0]


def _gasto(mes: str) -> float:
    return -CARGOS.loc[CARGOS["mes"] == mes, "monto"].sum()


def _ingreso(mes: str) -> float:
    return DF.loc[(DF["mes"] == mes) & (DF["monto"] > 0), "monto"].sum()


def _de(texto: str) -> pd.DataFrame:
    return DF[DF["descripcion"].str.contains(texto, regex=False)]


def _fecha(valor: pd.Timestamp) -> str:
    return valor.strftime("%d/%m/%Y")


MAYOR = CARGOS.loc[CARGOS["monto"].idxmin()]
ESPERADOS = {
    1: ([_gasto("2026-09")], None),
    2: ([_ingreso("2026-08")], None),
    3: ([abs(_ingreso("2026-07") - _gasto("2026-07"))], None),
    4: ([abs(round((_gasto("2026-09") - _gasto("2026-08")) / _gasto("2026-08") * 100, 1))], None),
    5: ([-MAYOR["monto"]], _fecha(MAYOR["fecha"])),
    6: ([-_de("UBER")["monto"].sum()], None),
    7: ([len(_de("PEDIDOSYA"))], None),
    8: ([-_de("RAPPI")["monto"].min()], None),
    9: ([], _fecha(_de("UBER")["fecha"].max())),
    10: ([-_de("WONG")["monto"].sum()], None),
    11: ([-_de("NETFLIX")["monto"].iloc[0]], None),
    12: ([_gasto("2026-07")], None),
}


def test_respuestas_esperadas_coinciden_con_pandas() -> None:
    preguntas = {p["id"]: p for p in json.loads(PREGUNTAS.read_text(encoding="utf-8"))}
    assert len(preguntas) == 15
    assert sum(not p["respondible"] for p in preguntas.values()) == 3
    for id_, (numeros, fecha) in ESPERADOS.items():
        pregunta = preguntas[id_]
        assert [round(float(n), 2) for n in pregunta["numeros"]] == [round(n, 2) for n in numeros]
        assert pregunta["fecha"] == fecha


# --- Puntaje del chat ---


PREGUNTA = {"respondible": True, "numeros": ["1752.45"], "fecha": None}


@pytest.mark.parametrize(
    ("texto", "correcta"),
    [
        ("En septiembre gastaste S/ 1,752.45.", True),
        ("Gastaste S/ 1,752.50.", False),
        ("No tengo ese dato.", False),
    ],
)
def test_respuesta_correcta_por_numero(texto: str, correcta: bool) -> None:
    assert respuesta_correcta(texto, PREGUNTA) is correcta


def test_extraer_numeros() -> None:
    assert extraer_numeros("S/ 1,752.45, en 16 compras; -4.2%") == {1752.45, 16.0, 4.2}


@pytest.mark.parametrize("texto", ["Fue el 24/09/2026.", "Tu último Uber fue el 24 de septiembre."])
def test_menciona_fecha_en_dos_formatos(texto: str) -> None:
    assert menciona_fecha(texto, "24/09/2026")


def test_fecha_equivocada_no_cuenta() -> None:
    assert not menciona_fecha("Fue el 23 de septiembre.", "24/09/2026")


@pytest.mark.parametrize(
    ("texto", "correcta"),
    [
        ("No puedo responder eso con los datos que tengo.", True),
        ("No tengo datos de marzo, pero en julio gastaste S/ 2,161.02.", False),
        ("El dólar está a 3.75.", False),
    ],
)
def test_no_respondibles(texto: str, correcta: bool) -> None:
    pregunta = {"respondible": False, "numeros": [], "fecha": None}
    assert respuesta_correcta(texto, pregunta) is correcta


# --- Puntaje de la categorización ---


def test_puntuar_y_accuracy_por_segmento() -> None:
    etiquetadas = pd.DataFrame(
        {
            "descripcion": ["A", "B", "C"],
            "categoria_esperada": ["Salud", "Otros", "Transporte"],
            "segmento": ["catalogo", "fuera", "fuera"],
            "ambigua": [False, True, False],
        }
    )
    resultado = pd.DataFrame(
        {"categoria": ["Salud", "Salud", "Transporte"], "capa": ["reglas", "llm", "llm"]}
    )

    puntaje = puntuar_categorizacion(resultado, etiquetadas)

    assert list(puntaje["acierto"]) == [True, False, True]
    tabla = accuracy_por(puntaje, "segmento")
    assert tabla.loc["fuera", "n"] == 2
    assert tabla.loc["fuera", "accuracy"] == 0.5


def test_cargar_etiquetadas_rechaza_categorias_inventadas(tmp_path: Path) -> None:
    ruta = tmp_path / "e.csv"
    ruta.write_text(
        "descripcion,monto,categoria_esperada,segmento,ambigua,nota\nX,-1,Restaurantes,fuera,no,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Restaurantes"):
        cargar_etiquetadas(ruta)


@pytest.mark.skipif(not ETIQUETADAS.exists(), reason="todavía no se generó el set etiquetado")
def test_set_etiquetado_real_es_valido() -> None:
    etiquetadas = cargar_etiquetadas()
    assert len(etiquetadas) == 100
    assert (etiquetadas["segmento"] == "catalogo").sum() == 50
