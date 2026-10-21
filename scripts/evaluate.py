"""Evalúa la categorización y el chat contra datos etiquetados. Usa la API real (cuesta).

- Categorización: data/eval/transacciones_etiquetadas.csv (etiquetado por una IA distinta a
  la que categoriza y revisado a mano). Accuracy global, por segmento, por capa y por categoría.
- Chat: data/eval/preguntas_chat.json sobre el archivo de ejemplo. Una respuesta es correcta si
  contiene los números (y la fecha) esperados; las no respondibles, si se niega sin dar cifras.

Uso: uv run python scripts/evaluate.py [--repeticiones 3] [--solo categorizacion|chat]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.categorize import CacheCategorias, categorizar  # noqa: E402
from src.chat import responder  # noqa: E402
from src.ingest import cargar_csv, limpiar_descripcion  # noqa: E402
from src.insights import MESES_ES, agregar_monto_pen, tipo_cambio_usd  # noqa: E402
from src.llm import ClienteLLM  # noqa: E402
from src.schemas import Categoria  # noqa: E402

ETIQUETADAS = Path("data/eval/transacciones_etiquetadas.csv")
PREGUNTAS = Path("data/eval/preguntas_chat.json")
EJEMPLO = Path("data/sample/estado_cuenta_ejemplo.csv")
RESULTADOS = Path("data/eval/resultados.md")
COLUMNAS = ["descripcion", "monto", "categoria_esperada", "segmento", "ambigua"]
SEGMENTOS = {"catalogo", "fuera"}
NEGATIVAS = ("no puedo", "no tengo", "no dispongo", "no es posible", "no cuento")
_NUMERO_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


# --- Datos ---


def cargar_etiquetadas(ruta: Path = ETIQUETADAS) -> pd.DataFrame:
    """Lee y valida el set etiquetado; falla con un mensaje claro si algo no cuadra."""
    df = pd.read_csv(ruta, dtype={"descripcion": str}, keep_default_na=False)
    faltantes = [c for c in COLUMNAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en {ruta}: {faltantes}")
    invalidas = set(df["categoria_esperada"]) - {c.value for c in Categoria}
    if invalidas:
        raise ValueError(f"Categorías que no existen: {sorted(invalidas)}")
    if not set(df["segmento"]) <= SEGMENTOS:
        raise ValueError(f"Segmentos válidos: {sorted(SEGMENTOS)}")
    df["monto"] = df["monto"].astype(float)
    df["ambigua"] = df["ambigua"].str.strip().str.lower().isin({"si", "sí"})
    return df


def a_formato_categorize(etiquetadas: pd.DataFrame) -> pd.DataFrame:
    """Normaliza las descripciones igual que ingest, sin pasar por fechas."""
    return pd.DataFrame(
        {
            "id": range(len(etiquetadas)),
            "descripcion": etiquetadas["descripcion"].map(limpiar_descripcion),
            "monto": etiquetadas["monto"],
        }
    )


# --- Puntaje ---


def puntuar_categorizacion(resultado: pd.DataFrame, etiquetadas: pd.DataFrame) -> pd.DataFrame:
    """Une predicción y etiqueta: una fila por transacción con `acierto`."""
    return etiquetadas.assign(
        prediccion=resultado["categoria"].to_numpy(),
        capa=resultado["capa"].to_numpy(),
        acierto=resultado["categoria"].to_numpy() == etiquetadas["categoria_esperada"].to_numpy(),
    )


def accuracy_por(puntaje: pd.DataFrame, columna: str) -> pd.DataFrame:
    tabla = puntaje.groupby(columna)["acierto"].agg(n="size", accuracy="mean")
    return tabla.sort_values("n", ascending=False)


def extraer_numeros(texto: str) -> set[float]:
    return {round(float(n.replace(",", "")), 2) for n in _NUMERO_RE.findall(texto)}


def menciona_fecha(texto: str, fecha: str) -> bool:
    """Acepta "24/09/2026" o "24 de septiembre"."""
    dia, mes, _ = (int(p) for p in fecha.split("/"))
    texto = texto.lower()
    return fecha in texto or bool(re.search(rf"\b{dia}\b[^.\n]*{MESES_ES[mes - 1]}", texto))


def respuesta_correcta(texto: str, pregunta: dict) -> bool:
    if not pregunta["respondible"]:
        texto_min = texto.lower()
        return any(n in texto_min for n in NEGATIVAS) and "s/" not in texto_min
    esperados = {round(float(n), 2) for n in pregunta["numeros"]}
    if not esperados <= extraer_numeros(texto):
        return False
    return pregunta["fecha"] is None or menciona_fecha(texto, pregunta["fecha"])


# --- Corridas ---


@dataclass
class Corrida:
    global_: float
    por_segmento: pd.DataFrame
    por_capa: pd.DataFrame
    por_categoria: pd.DataFrame
    sin_ambiguas: float
    puntaje: pd.DataFrame


def evaluar_categorizacion(etiquetadas: pd.DataFrame, llm: ClienteLLM) -> Corrida:
    resultado, _ = categorizar(a_formato_categorize(etiquetadas), CacheCategorias(), llm)
    puntaje = puntuar_categorizacion(resultado, etiquetadas)
    return Corrida(
        global_=puntaje["acierto"].mean(),
        por_segmento=accuracy_por(puntaje, "segmento"),
        por_capa=accuracy_por(puntaje, "capa"),
        por_categoria=accuracy_por(puntaje, "categoria_esperada"),
        sin_ambiguas=puntaje.loc[~puntaje["ambigua"], "acierto"].mean(),
        puntaje=puntaje,
    )


def evaluar_chat(preguntas: list[dict], llm: ClienteLLM) -> list[dict]:
    df, _ = categorizar(cargar_csv(EJEMPLO).df, CacheCategorias(), llm)
    df = agregar_monto_pen(df, tipo_cambio_usd())
    filas = []
    for pregunta in preguntas:
        respuesta = responder(pregunta["pregunta"], [], df, llm)
        filas.append(
            {
                "id": pregunta["id"],
                "pregunta": pregunta["pregunta"],
                "correcta": respuesta_correcta(respuesta.texto, pregunta),
                "herramientas": ", ".join(respuesta.herramientas) or "-",
                "respuesta": respuesta.texto,
            }
        )
    return filas


# --- Reporte ---


def _pct(valor: float) -> str:
    return f"{valor:.1%}"


def _rango(valores: list[float]) -> str:
    return f"{_pct(sum(valores) / len(valores))} (rango {_pct(min(valores))}–{_pct(max(valores))})"


def reporte_categorizacion(corridas: list[Corrida]) -> str:
    primera = corridas[0]
    lineas = [
        "## Categorización",
        "",
        f"- Transacciones: {len(primera.puntaje)} "
        f"({(primera.puntaje['segmento'] == 'catalogo').sum()} del catálogo, "
        f"{(primera.puntaje['segmento'] == 'fuera').sum()} fuera del catálogo; "
        f"{primera.puntaje['ambigua'].sum()} marcadas como ambiguas)",
        f"- Corridas: {len(corridas)}",
        "",
        "| Métrica | Accuracy |",
        "|---|---|",
        f"| Global | {_rango([c.global_ for c in corridas])} |",
        f"| Global sin ambiguas | {_rango([c.sin_ambiguas for c in corridas])} |",
    ]
    for segmento in primera.por_segmento.index:
        valores = [c.por_segmento.loc[segmento, "accuracy"] for c in corridas]
        n = primera.por_segmento.loc[segmento, "n"]
        lineas.append(f"| Segmento «{segmento}» (n={n}) | {_rango(valores)} |")
    for capa in primera.por_capa.index:
        valores = [c.por_capa.loc[capa, "accuracy"] for c in corridas if capa in c.por_capa.index]
        n = primera.por_capa.loc[capa, "n"]
        lineas.append(f"| Capa «{capa}» (n={n}, 1.ª corrida) | {_rango(valores)} |")

    lineas += [
        "",
        "**Por categoría (1.ª corrida)**",
        "",
        "| Categoría | n | Accuracy |",
        "|---|---|---|",
    ]
    for categoria, fila in primera.por_categoria.iterrows():
        lineas.append(f"| {categoria} | {int(fila['n'])} | {_pct(fila['accuracy'])} |")

    errores = primera.puntaje[~primera.puntaje["acierto"]]
    lineas += [
        "",
        "**Errores (1.ª corrida)**",
        "",
        "| Descripción | Esperada | Predicha | Capa | Ambigua |",
        "|---|---|---|---|---|",
    ]
    for fila in errores.itertuples():
        ambigua = "sí" if fila.ambigua else "no"
        lineas.append(
            f"| {fila.descripcion} | {fila.categoria_esperada} | {fila.prediccion} | {fila.capa} | "
            f"{ambigua} |"
        )
    return "\n".join(lineas)


def reporte_chat(corridas: list[list[dict]]) -> str:
    aciertos = [sum(f["correcta"] for f in corrida) / len(corrida) for corrida in corridas]
    lineas = [
        "## Chat",
        "",
        f"- Preguntas: {len(corridas[0])} (incluye 3 que no se pueden responder) · "
        f"corridas: {len(corridas)}",
        f"- Respuestas correctas: {_rango(aciertos)}",
        "",
        "| # | Pregunta | Correctas | Herramientas (1.ª corrida) |",
        "|---|---|---|---|",
    ]
    for i, fila in enumerate(corridas[0]):
        correctas = sum(corrida[i]["correcta"] for corrida in corridas)
        lineas.append(
            f"| {fila['id']} | {fila['pregunta']} | {correctas}/{len(corridas)} | "
            f"{fila['herramientas']} |"
        )
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--solo", choices=["categorizacion", "chat"])
    args = parser.parse_args()

    llm = ClienteLLM()
    secciones = [f"# Resultados de la evaluación ({date.today():%d/%m/%Y}, modelo {llm.modelo})"]

    if args.solo != "chat":
        if ETIQUETADAS.exists():
            etiquetadas = cargar_etiquetadas()
            corridas = [evaluar_categorizacion(etiquetadas, llm) for _ in range(args.repeticiones)]
            secciones.append(reporte_categorizacion(corridas))
        else:
            print(f"Falta {ETIQUETADAS}: se omite la categorización.")

    if args.solo != "categorizacion":
        preguntas = json.loads(PREGUNTAS.read_text(encoding="utf-8"))
        corridas_chat = [evaluar_chat(preguntas, llm) for _ in range(args.repeticiones)]
        secciones.append(reporte_chat(corridas_chat))

    costo = llm.uso.costo_usd(llm.modelo)
    secciones.append(
        f"_Costo total de la evaluación: US$ {costo:.4f} ({llm.uso.llamadas} llamadas, "
        f"{llm.uso.tokens_entrada:,} tokens de entrada, {llm.uso.tokens_salida:,} de salida)._"
    )
    reporte = "\n\n".join(secciones) + "\n"
    RESULTADOS.write_text(reporte, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(reporte)


if __name__ == "__main__":
    main()
