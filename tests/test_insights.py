import pandas as pd
import pytest

from src.insights import (
    agregar_monto_pen,
    evolucion_mensual,
    gasto_por_categoria,
    hubo_conversion,
    kpis_mes,
    meses_disponibles,
    top_comercios,
)


def _df(*filas: tuple[str, str, float, str, str]) -> pd.DataFrame:
    df = pd.DataFrame(filas, columns=["mes", "descripcion", "monto", "moneda", "categoria"])
    return agregar_monto_pen(df, tipo_cambio=3.75)


DATOS = _df(
    ("2026-08", "RAPPI PERU SAC", -100.0, "PEN", "Comida y delivery"),
    ("2026-08", "TRANSFERENCIA RECIBIDA", 500.0, "PEN", "Transferencias"),
    ("2026-09", "RAPPI PERU SAC", -80.0, "PEN", "Comida y delivery"),
    ("2026-09", "RAPPI PERU SAC", -40.0, "PEN", "Comida y delivery"),
    ("2026-09", "POS WONG", -60.0, "PEN", "Supermercado"),
    ("2026-09", "YAPE-ANA TORRES", -200.0, "PEN", "Transferencias"),
    ("2026-09", "ABONO PLANILLA", 1000.0, "PEN", "Ingresos"),
)


def test_convierte_dolares_y_conserva_el_original() -> None:
    df = _df(("2026-09", "OPENAI", -20.0, "USD", "Suscripciones"))
    assert df.loc[0, "monto_pen"] == -75.0
    assert df.loc[0, "monto"] == -20.0
    assert hubo_conversion(df)
    assert not hubo_conversion(DATOS)


def test_meses_disponibles_ordenados() -> None:
    assert meses_disponibles(DATOS) == ["2026-08", "2026-09"]


def test_kpis_por_signo_no_por_categoria() -> None:
    kpis = kpis_mes(DATOS, "2026-09")
    assert kpis.gasto == 380.0  # incluye el Yape que sale
    assert kpis.ingreso == 1000.0
    assert kpis.balance == 620.0


def test_transferencia_recibida_cuenta_como_ingreso() -> None:
    assert kpis_mes(DATOS, "2026-08").ingreso == 500.0


def test_variacion_contra_el_mes_anterior() -> None:
    assert kpis_mes(DATOS, "2026-09").variacion_gasto == pytest.approx(2.8)  # 100 → 380


def test_primer_mes_no_tiene_variacion() -> None:
    assert kpis_mes(DATOS, "2026-08").variacion_gasto is None


def test_gasto_por_categoria_ordenado() -> None:
    serie = gasto_por_categoria(DATOS, "2026-09")
    assert list(serie.index) == ["Transferencias", "Comida y delivery", "Supermercado"]
    assert list(serie) == [200.0, 120.0, 60.0]


def test_evolucion_mensual() -> None:
    evolucion = evolucion_mensual(DATOS)
    assert list(evolucion.index) == ["2026-08", "2026-09"]
    assert evolucion.loc["2026-08", "balance"] == 400.0


def test_top_comercios_sin_transferencias() -> None:
    top = top_comercios(DATOS, "2026-09")
    assert list(top.index) == ["RAPPI PERU SAC", "POS WONG"]
    assert top["RAPPI PERU SAC"] == 120.0


def test_top_limita_a_n() -> None:
    filas = [("2026-09", f"COMERCIO {i}", -float(i + 1), "PEN", "Otros") for i in range(8)]
    assert len(top_comercios(_df(*filas), "2026-09")) == 5
