import pandas as pd
import pytest

from src.ingest import DESCRIPCION_VACIA, IngestError, cargar_csv, limpiar_descripcion


def _csv(texto: str) -> bytes:
    return texto.strip().encode("utf-8")


def test_contrato_de_salida() -> None:
    resultado = cargar_csv(_csv("fecha,descripcion,monto\n15/09/2026,Rappi*1234,-35.50"))

    assert list(resultado.df.columns) == [
        "id",
        "fecha",
        "mes",
        "descripcion_original",
        "descripcion",
        "monto",
        "moneda",
    ]
    fila = resultado.df.iloc[0]
    assert fila["fecha"] == pd.Timestamp(2026, 9, 15)
    assert fila["mes"] == "2026-09"
    assert fila["descripcion"] == "RAPPI"
    assert fila["monto"] == -35.50
    assert fila["moneda"] == "PEN"


def test_falta_columna_obligatoria() -> None:
    with pytest.raises(IngestError, match="monto"):
        cargar_csv(_csv("fecha,descripcion\n15/09/2026,Rappi"))


def test_archivo_vacio() -> None:
    with pytest.raises(IngestError, match="vacío"):
        cargar_csv(b"")


def test_columnas_con_mayusculas_y_espacios() -> None:
    resultado = cargar_csv(_csv(" Fecha ,DESCRIPCION,Monto\n15/09/2026,Wong,-80"))
    assert len(resultado.df) == 1


def test_fechas_ambiguas_asumen_formato_peruano() -> None:
    resultado = cargar_csv(_csv("fecha,descripcion,monto\n03/04/2026,Wong,-10"))
    assert resultado.df.iloc[0]["fecha"] == pd.Timestamp(2026, 4, 3)


def test_detecta_formato_mm_dd_por_el_archivo() -> None:
    texto = "fecha,descripcion,monto\n03/04/2026,Wong,-10\n09/25/2026,Tottus,-20"
    df = cargar_csv(_csv(texto)).df
    assert list(df["fecha"]) == [pd.Timestamp(2026, 3, 4), pd.Timestamp(2026, 9, 25)]


def test_fechas_mezcladas_son_error() -> None:
    texto = "fecha,descripcion,monto\n25/09/2026,Wong,-10\n09/25/2026,Tottus,-20"
    with pytest.raises(IngestError, match="mezclan"):
        cargar_csv(_csv(texto))


def test_anio_de_dos_digitos() -> None:
    resultado = cargar_csv(_csv("fecha,descripcion,monto\n15/09/26,Wong,-10"))
    assert resultado.df.iloc[0]["fecha"] == pd.Timestamp(2026, 9, 15)


@pytest.mark.parametrize(
    ("monto", "esperado", "moneda"),
    [
        ("-1234.50", -1234.50, "PEN"),
        ("-1,234.50", -1234.50, "PEN"),
        ("S/ -50", -50.0, "PEN"),
        ("S/.50", 50.0, "PEN"),
        ("-$12", -12.0, "USD"),
        ("US$ 12.99", 12.99, "USD"),
    ],
)
def test_formatos_de_monto(monto: str, esperado: float, moneda: str) -> None:
    resultado = cargar_csv(_csv(f'fecha,descripcion,monto\n15/09/2026,X,"{monto}"'))
    fila = resultado.df.iloc[0]
    assert fila["monto"] == esperado
    assert fila["moneda"] == moneda


def test_sin_columna_moneda_asume_pen_y_avisa() -> None:
    resultado = cargar_csv(_csv("fecha,descripcion,monto\n15/09/2026,Wong,-10"))
    assert any("columna 'moneda'" in aviso for aviso in resultado.avisos)


def test_simbolo_gana_sobre_columna_moneda() -> None:
    texto = "fecha,descripcion,monto,moneda\n15/09/2026,Netflix,$-10,PEN"
    resultado = cargar_csv(_csv(texto))
    assert resultado.df.iloc[0]["moneda"] == "USD"
    assert any("no coincide" in aviso for aviso in resultado.avisos)


def test_columna_moneda_vacia_asume_pen() -> None:
    texto = "fecha,descripcion,monto,moneda\n15/09/2026,Wong,-10,\n16/09/2026,Uber,-5,USD"
    resultado = cargar_csv(_csv(texto))
    assert list(resultado.df["moneda"]) == ["PEN", "USD"]
    assert any("se asumió PEN (líneas 2)" in aviso for aviso in resultado.avisos)


def test_elimina_filas_sin_fecha_o_monto_y_avisa_lineas() -> None:
    texto = (
        "fecha,descripcion,monto\n"
        "15/09/2026,Wong,-10\n"  # línea 2: válida
        ",Tottus,-20\n"  # línea 3: sin fecha
        "31/02/2026,Uber,-5\n"  # línea 4: fecha imposible
        "16/09/2026,Rappi,\n"  # línea 5: sin monto
        ",,\n"  # línea 6: vacía
    )
    resultado = cargar_csv(_csv(texto))

    assert list(resultado.df["descripcion"]) == ["WONG"]
    assert "Se eliminaron 2 filas sin fecha válida (líneas 3, 4)." in resultado.avisos
    assert any(a.startswith("Se eliminaron 1 filas vacías (líneas 6)") for a in resultado.avisos)


def test_csv_mal_formado_es_error() -> None:
    texto = "fecha,descripcion,monto\n15/09/2026,Plin,1,234.50"
    with pytest.raises(IngestError, match="mal formado"):
        cargar_csv(_csv(texto))


def test_monto_ilegible_se_elimina() -> None:
    texto = 'fecha,descripcion,monto\n15/09/2026,Wong,-10\n16/09/2026,Rappi,"1.234,50"'
    resultado = cargar_csv(_csv(texto))
    assert len(resultado.df) == 1
    assert "Se eliminaron 1 filas sin monto válido (líneas 3)." in resultado.avisos


def test_sin_filas_validas_es_error() -> None:
    with pytest.raises(IngestError, match="ninguna transacción"):
        cargar_csv(_csv("fecha,descripcion,monto\n,Wong,-10"))


def test_descripcion_vacia_se_conserva() -> None:
    resultado = cargar_csv(_csv("fecha,descripcion,monto\n15/09/2026,,-10"))
    assert resultado.df.iloc[0]["descripcion"] == DESCRIPCION_VACIA
    assert any("no tienen descripción" in aviso for aviso in resultado.avisos)


def test_separador_punto_y_coma_y_latin1() -> None:
    datos = "fecha;descripcion;monto\n15/09/2026;Pensión UPC;-1500\n".encode("latin-1")
    resultado = cargar_csv(datos)
    assert resultado.df.iloc[0]["descripcion"] == "PENSIÓN UPC"


def test_ordena_por_fecha_e_id_correlativo() -> None:
    texto = "fecha,descripcion,monto\n20/09/2026,B,-1\n10/09/2026,A,-1"
    df = cargar_csv(_csv(texto)).df
    assert list(df["descripcion"]) == ["A", "B"]
    assert list(df["id"]) == [0, 1]


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("Rappi*1234", "RAPPI"),
        ("  uber   trip  ", "UBER TRIP"),
        ("PAGO PENSION UPC 202609", "PAGO PENSION UPC"),
        ("YAPE A JUAN P REF:AB12", "YAPE A JUAN P"),
        ("NETFLIX.COM #998", "NETFLIX.COM"),
        ("*1234", DESCRIPCION_VACIA),
    ],
)
def test_limpiar_descripcion(entrada: str, esperado: str) -> None:
    assert limpiar_descripcion(entrada) == esperado
