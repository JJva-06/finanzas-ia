from pathlib import Path

import pandas as pd

from scripts.generate_sample import FIJOS, SALIDA_POR_DEFECTO, generar, guardar
from src.ingest import cargar_csv


def test_misma_semilla_mismo_resultado() -> None:
    df_a, perfil_a = generar(7)
    df_b, perfil_b = generar(7)
    pd.testing.assert_frame_equal(df_a, df_b)
    assert perfil_a == perfil_b


def test_semillas_distintas_dan_datos_distintos() -> None:
    assert not generar(1)[0].equals(generar(2)[0])


def test_volumen_y_periodo() -> None:
    df, _ = generar()
    assert 120 <= len(df) <= 180
    meses = {(f.year, f.month) for f in df["fecha"]}
    assert meses == {(2026, 7), (2026, 8), (2026, 9)}


def test_el_csv_pasa_por_ingest_sin_avisos(tmp_path: Path) -> None:
    df, _ = generar()
    ruta = tmp_path / "muestra.csv"
    guardar(df, ruta)

    resultado = cargar_csv(ruta)
    assert resultado.avisos == []
    assert len(resultado.df) == len(df)
    assert set(resultado.df["moneda"]) == {"PEN"}


def test_suscripciones_con_monto_fijo_cada_mes() -> None:
    df, _ = generar()
    for fijo in FIJOS:
        cargos = df[df["descripcion"] == fijo.descripcion]
        assert len(cargos) == 3
        assert set(cargos["monto"]) == {-fijo.monto}


def test_ambos_perfiles_tienen_ingresos() -> None:
    perfiles_vistos = set()
    for seed in range(20):
        df, perfil = generar(seed)
        perfiles_vistos.add(perfil)
        assert (df["monto"] > 0).sum() >= 3
    assert perfiles_vistos == {"dependiente", "independiente"}


def test_archivo_del_repo_coincide_con_la_semilla(tmp_path: Path) -> None:
    """Si falla, alguien cambió el generador sin regenerar el archivo de ejemplo."""
    ruta = tmp_path / "regenerado.csv"
    guardar(generar()[0], ruta)
    regenerado = ruta.read_text(encoding="utf-8").splitlines()
    assert regenerado == SALIDA_POR_DEFECTO.read_text(encoding="utf-8").splitlines()
