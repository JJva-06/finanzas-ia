"""Genera un estado de cuenta SINTÉTICO de 3 meses para un estudiante en Lima.

Todos los nombres de personas y negocios locales son inventados; nunca usar datos reales.
La semilla hace que el archivo sea reproducible: la demo, el README y la evaluación
dependen de que este archivo no cambie.

Uso: uv run python scripts/generate_sample.py [--seed 42] [--salida ruta.csv]
"""

from __future__ import annotations

import argparse
import calendar
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

MESES = ((2026, 7), (2026, 8), (2026, 9))
SEMILLA_POR_DEFECTO = 42
SALIDA_POR_DEFECTO = Path("data/sample/estado_cuenta_ejemplo.csv")
PERFILES = ("dependiente", "independiente")


@dataclass(frozen=True)
class Variable:
    """Comercio con monto y frecuencia aleatorios; `conocido` = lo reconocerán las reglas."""

    descripcion: str
    minimo: float
    maximo: float
    veces_por_mes: tuple[int, int]
    conocido: bool = True
    con_codigo: bool = True


@dataclass(frozen=True)
class Fijo:
    """Cargo recurrente: mismo monto, mismo día, cada mes."""

    descripcion: str
    monto: float
    dia: int
    moneda: str = "PEN"


@dataclass(frozen=True)
class Desconocido:
    """Negocio local sin regla: la primera aparición la resuelve el LLM y el resto la caché."""

    descripcion: str
    minimo: float
    maximo: float
    veces_total: tuple[int, int]


VARIABLES = (
    Variable("RAPPI*PERU SAC", 15, 60, (4, 7)),
    Variable("PEDIDOSYA*PEDIDO", 12, 50, (2, 4)),
    Variable("UBER *TRIP", 8, 30, (4, 7)),
    Variable("INDRIVE*VIAJE", 7, 20, (1, 3)),
    Variable("POS WONG BENAVIDES", 40, 180, (1, 2)),
    Variable("POS PLAZA VEA SURCO", 30, 150, (1, 3)),
    Variable("TOTTUS JOCKEY PLAZA", 25, 120, (0, 2)),
    Variable("TAMBO+ LARCO", 4, 25, (1, 4)),
    Variable("MERPAGO*CINEPLANET", 15, 45, (0, 2)),
    Variable("IZI*INKAFARMA", 10, 80, (0, 2)),
    Variable("MIFARMA MIRAFLORES", 8, 60, (0, 1)),
    Variable("YAPE-ANA TORRES", 10, 80, (1, 3), con_codigo=False),
    Variable("YAPE-LUIS PAREDES", 10, 60, (0, 2), con_codigo=False),
    Variable("PLIN-CARLA MENDOZA", 10, 50, (0, 2), con_codigo=False),
)

FIJOS = (
    Fijo("DLC*NETFLIX.COM", 44.90, 5),
    Fijo("PAYPAL *SPOTIFY", 21.90, 12),
    Fijo("RAPPI*RAPPI PRO", 24.90, 3),
    Fijo("OPENAI *CHATGPT SUBSCR", 20.00, 18, moneda="USD"),
    Fijo("MOVISTAR MOVIL PLAN POSTPAGO", 59.90, 20),
)
PENSION = Fijo("PAGO PENSION UNIVERSIDAD", 950.00, 8)

DESCONOCIDOS = (
    Desconocido("IZI*CHIFA DRAGON DORADO", 18, 45, (4, 7)),
    Desconocido("BODEGA DON LUCHO", 3, 25, (4, 7)),
    Desconocido("MERPAGO*LIBRERIA EL ESTUDIANTE", 15, 90, (4, 6)),
    Desconocido("IZI*POLLERIA LA BRASA", 25, 60, (4, 6)),
    Desconocido("NIUBIZ*BARBERIA EL CORTE", 20, 35, (4, 5)),
    Desconocido("VENDEMAS*JUGUERIA LA FRUTA", 6, 15, (4, 7)),
    Desconocido("CULQI*GIMNASIO FORMA FIT", 60, 90, (4, 4)),
    Desconocido("IZI*COPIAS Y MAS", 2, 12, (4, 7)),
)


def generar(seed: int = SEMILLA_POR_DEFECTO) -> tuple[pd.DataFrame, str]:
    """Devuelve las transacciones ordenadas por fecha y el perfil elegido."""
    rng = random.Random(seed)
    perfil = rng.choice(PERFILES)

    filas = _ingresos(rng, perfil) + _fijos(perfil) + _variables(rng) + _desconocidos(rng)
    df = pd.DataFrame(filas, columns=["fecha", "descripcion", "monto", "moneda", "conocido"])
    df = df.sort_values(["fecha", "descripcion"], kind="stable").reset_index(drop=True)
    return df, perfil


def guardar(df: pd.DataFrame, salida: Path) -> None:
    """Escribe el CSV con el formato que espera `ingest` (sin la columna interna `conocido`)."""
    salida.parent.mkdir(parents=True, exist_ok=True)
    exportar = df.drop(columns="conocido").assign(fecha=df["fecha"].map(_formato_peruano))
    exportar.to_csv(salida, index=False, float_format="%.2f", lineterminator="\n")


def _ingresos(rng: random.Random, perfil: str) -> list[tuple]:
    if perfil == "dependiente":
        # Propina de los padres cada 1 o 2 semanas.
        cada = rng.choice((7, 14))
        monto = 450.0 if cada == 7 else 900.0
        dia = date(*MESES[0], rng.randint(1, 7))
        filas = []
        while dia <= _ultimo_dia(MESES[-1]):
            filas.append((dia, "TRANSFERENCIA RECIBIDA M QUISPE", monto, "PEN", True))
            dia += timedelta(days=cada)
        return filas

    # Independiente: sueldo de prácticas, mensual o quincenal.
    quincenal = rng.random() < 0.5
    filas = []
    for anio, mes in MESES:
        fin = _ultimo_dia((anio, mes))
        if quincenal:
            filas.append(
                (date(anio, mes, 15), "ABONO PLANILLA INNOVA DATA SAC", 1500.0, "PEN", True)
            )
            filas.append((fin, "ABONO PLANILLA INNOVA DATA SAC", 1500.0, "PEN", True))
        else:
            filas.append((fin, "ABONO PLANILLA INNOVA DATA SAC", 3000.0, "PEN", True))
    return filas


def _fijos(perfil: str) -> list[tuple]:
    # En el perfil dependiente, los padres pagan la pensión fuera de esta cuenta.
    fijos = FIJOS + ((PENSION,) if perfil == "independiente" else ())
    return [
        (date(anio, mes, f.dia), f.descripcion, -f.monto, f.moneda, True)
        for anio, mes in MESES
        for f in fijos
    ]


def _variables(rng: random.Random) -> list[tuple]:
    filas = []
    for anio, mes in MESES:
        for v in VARIABLES:
            for _ in range(rng.randint(*v.veces_por_mes)):
                descripcion = _con_ruido(rng, v.descripcion) if v.con_codigo else v.descripcion
                monto = -round(rng.uniform(v.minimo, v.maximo), 2)
                filas.append((_dia_al_azar(rng, anio, mes), descripcion, monto, "PEN", v.conocido))
    return filas


def _desconocidos(rng: random.Random) -> list[tuple]:
    filas = []
    for d in DESCONOCIDOS:
        for _ in range(rng.randint(*d.veces_total)):
            anio, mes = rng.choice(MESES)
            monto = -round(rng.uniform(d.minimo, d.maximo), 2)
            filas.append((_dia_al_azar(rng, anio, mes), d.descripcion, monto, "PEN", False))
    return filas


def _con_ruido(rng: random.Random, descripcion: str) -> str:
    """Agrega un n° de operación variable, como en un estado de cuenta real."""
    return f"{descripcion} {rng.randint(1000, 99999)}"


def _dia_al_azar(rng: random.Random, anio: int, mes: int) -> date:
    return date(anio, mes, rng.randint(1, calendar.monthrange(anio, mes)[1]))


def _ultimo_dia(anio_mes: tuple[int, int]) -> date:
    anio, mes = anio_mes
    return date(anio, mes, calendar.monthrange(anio, mes)[1])


def _formato_peruano(dia: date) -> str:
    return dia.strftime("%d/%m/%Y")


def _resumen(df: pd.DataFrame, perfil: str) -> str:
    conocidos = df["conocido"].mean()
    desconocidos = df.loc[~df["conocido"], "descripcion"]
    return (
        f"Perfil: {perfil}\n"
        f"Transacciones: {len(df)}\n"
        f"Comercios conocidos (reglas): {conocidos:.0%}\n"
        f"Desconocidos: {len(desconocidos)} filas, {desconocidos.nunique()} comercios distintos "
        f"(~{desconocidos.nunique() / len(df):.0%} LLM, "
        f"~{(len(desconocidos) - desconocidos.nunique()) / len(df):.0%} caché)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEMILLA_POR_DEFECTO)
    parser.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    args = parser.parse_args()

    df, perfil = generar(args.seed)
    guardar(df, args.salida)
    print(f"Archivo generado: {args.salida}\n{_resumen(df, perfil)}")


if __name__ == "__main__":
    main()
