import pandas as pd
import pytest

from src.categorize import CacheCategorias, categorizar
from src.ingest import cargar_csv
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
    redactar_con_plantilla,
    top_comercios,
)


def _df(*filas: tuple[str, str, float, str, str]) -> pd.DataFrame:
    df = pd.DataFrame(filas, columns=["mes", "descripcion", "monto", "moneda", "categoria"])
    return agregar_monto_pen(df, tipo_cambio=3.75)


SAMPLE = "data/sample/estado_cuenta_ejemplo.csv"

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


def _suscripciones(*filas: tuple[str, str, float, str, str]) -> dict[str, str]:
    """descripcion → origen de cada suscripción detectada."""
    detectadas = detectar_suscripciones(_df(*filas))
    return dict(zip(detectadas["descripcion"], detectadas["origen"], strict=True))


def test_mismo_monto_cada_mes_es_suscripcion_por_patron() -> None:
    meses = ["2026-07", "2026-08", "2026-09"]
    filas = [
        (m, "MOVISTAR MOVIL", -59.9, "PEN", "Servicios (luz, agua, internet, celular)")
        for m in meses
    ]
    detectadas = detectar_suscripciones(_df(*filas))

    assert _suscripciones(*filas) == {"MOVISTAR MOVIL": "patrón"}
    assert detectadas.loc[0, "costo_anual"] == pytest.approx(718.8)
    assert detectadas.loc[0, "meses"] == meses


def test_varios_cargos_por_mes_no_es_suscripcion() -> None:
    filas = [
        ("2026-08", "RAPPI PERU SAC", -30.0, "PEN", "Comida y delivery"),
        ("2026-08", "RAPPI PERU SAC", -30.0, "PEN", "Comida y delivery"),
        ("2026-09", "RAPPI PERU SAC", -30.0, "PEN", "Comida y delivery"),
    ]
    assert _suscripciones(*filas) == {}


@pytest.mark.parametrize(("segundo", "detectada"), [(-65.0, True), (-90.0, False)])
def test_tolerancia_del_diez_por_ciento(segundo: float, detectada: bool) -> None:
    filas = [
        ("2026-07", "CULQI GIMNASIO", -72.0, "PEN", "Salud"),
        ("2026-08", "CULQI GIMNASIO", segundo, "PEN", "Salud"),
    ]
    assert ("CULQI GIMNASIO" in _suscripciones(*filas)) == detectada


def test_meses_no_consecutivos_no_cuentan() -> None:
    filas = [
        ("2026-07", "CULQI GIMNASIO", -72.0, "PEN", "Salud"),
        ("2026-09", "CULQI GIMNASIO", -72.0, "PEN", "Salud"),
    ]
    assert _suscripciones(*filas) == {}


def test_categoria_suscripciones_entra_aunque_no_cumpla_el_patron() -> None:
    filas = [
        ("2026-09", "DISNEY", -39.9, "PEN", "Suscripciones"),
        ("2026-07", "PAYPAL SPOTIFY", -21.9, "PEN", "Suscripciones"),
        ("2026-08", "PAYPAL SPOTIFY", -26.9, "PEN", "Suscripciones"),
    ]
    assert _suscripciones(*filas) == {"DISNEY": "categoría", "PAYPAL SPOTIFY": "categoría"}


def test_solo_cargos() -> None:
    filas = [
        (m, "TRANSFERENCIA RECIBIDA", 450.0, "PEN", "Transferencias")
        for m in ["2026-07", "2026-08"]
    ]
    assert _suscripciones(*filas) == {}


def test_suscripciones_en_el_archivo_de_ejemplo() -> None:
    df, _ = categorizar(cargar_csv(SAMPLE).df, CacheCategorias(), None)
    detectadas = detectar_suscripciones(agregar_monto_pen(df, 3.75))
    origen = dict(zip(detectadas["descripcion"], detectadas["origen"], strict=True))

    for fijo in [
        "DLC NETFLIX.COM",
        "PAYPAL SPOTIFY",
        "RAPPI RAPPI PRO",
        "OPENAI CHATGPT SUBSCR",
        "MOVISTAR MOVIL PLAN POSTPAGO",
    ]:
        assert origen[fijo] == "patrón"
    assert "RAPPI PERU SAC" not in origen
    assert list(detectadas["costo_anual"]) == sorted(detectadas["costo_anual"], reverse=True)


def _hechos(*filas: tuple[str, str, float, str, str]) -> dict[str, dict[str, str]]:
    df = _df(*filas)
    return {h.tipo: h.datos for h in hechos_del_mes(df, max(df["mes"]))}


def test_formatear_soles() -> None:
    assert formatear_soles(1234.5) == "S/ 1,234.50"
    assert formatear_soles(-286.0) == "-S/ 286.00"


def test_hechos_del_mes_con_mes_anterior() -> None:
    hechos = _hechos(*DATOS.drop(columns="monto_pen").itertuples(index=False))

    assert hechos["ahorro"] == {
        "mes": "septiembre",
        "balance": "S/ 620.00",
        "mes_anterior": "agosto",
        "balance_anterior": "S/ 400.00",
    }
    # Transferencias creció más (0 → 200), pero no se nombra a la persona como comercio.
    assert hechos["categoria_crecio"]["categoria"] == "Transferencias"
    assert "comercio" not in hechos["categoria_crecio"]


def test_categoria_crecio_nombra_su_comercio_principal() -> None:
    hechos = _hechos(
        ("2026-08", "RAPPI PERU SAC", -50.0, "PEN", "Comida y delivery"),
        ("2026-09", "RAPPI PERU SAC", -150.0, "PEN", "Comida y delivery"),
        ("2026-09", "IZI CHIFA", -40.0, "PEN", "Comida y delivery"),
    )
    assert hechos["categoria_crecio"]["comercio"] == "RAPPI PERU SAC"
    assert hechos["categoria_crecio"]["gasto"] == "S/ 190.00"
    assert hechos["categoria_crecio"]["gasto_anterior"] == "S/ 50.00"


def test_primer_mes_usa_categoria_principal() -> None:
    hechos = _hechos(
        ("2026-09", "POS WONG", -75.0, "PEN", "Supermercado"),
        ("2026-09", "RAPPI PERU SAC", -25.0, "PEN", "Comida y delivery"),
    )
    assert "mes_anterior" not in hechos["ahorro"]
    assert hechos["categoria_principal"]["porcentaje"] == "75%"
    assert hechos["categoria_principal"]["comercio"] == "POS WONG"


def test_gasto_hormiga_sin_transferencias() -> None:
    hechos = _hechos(
        ("2026-09", "TAMBO LARCO", -5.0, "PEN", "Supermercado"),
        ("2026-09", "BODEGA", -12.5, "PEN", "Supermercado"),
        ("2026-09", "YAPE-ANA TORRES", -10.0, "PEN", "Transferencias"),
        ("2026-09", "POS WONG", -80.0, "PEN", "Supermercado"),
    )
    assert hechos["hormiga"]["cantidad"] == "2"
    assert hechos["hormiga"]["total"] == "S/ 17.50"


def test_sin_compras_chicas_no_hay_hecho_hormiga() -> None:
    hechos = _hechos(("2026-09", "POS WONG", -80.0, "PEN", "Supermercado"))
    assert "hormiga" not in hechos


def test_plantillas_de_todos_los_tipos_en_el_ejemplo() -> None:
    df, _ = categorizar(cargar_csv(SAMPLE).df, CacheCategorias(), None)
    df = agregar_monto_pen(df, 3.75)
    for mes in meses_disponibles(df):
        for hecho in hechos_del_mes(df, mes):
            texto = redactar_con_plantilla(hecho)
            assert texto.endswith(".") and "{" not in texto
