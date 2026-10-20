"""Carga y normalización de estados de cuenta en CSV.

Contrato de salida (`IngestResult.df`), una fila por transacción válida:
    id                    int        correlativo, ordenado por fecha
    fecha                 datetime64
    mes                   str        "YYYY-MM"
    descripcion_original  str
    descripcion           str        normalizada (mayúsculas, sin códigos de referencia)
    monto                 float      negativo = cargo, positivo = abono
    moneda                str        "PEN" o "USD"
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd

COLUMNAS_REQUERIDAS = ("fecha", "descripcion", "monto")
MONEDA_POR_DEFECTO = "PEN"
DESCRIPCION_VACIA = "SIN DESCRIPCION"
MAX_LINEAS_EN_AVISO = 10

# El orden importa: "US$" antes que "$" y "S/." antes que "S/".
_SIMBOLOS_MONEDA = (
    ("US$", "USD"),
    ("USD", "USD"),
    ("$", "USD"),
    ("S/.", "PEN"),
    ("S/", "PEN"),
    ("PEN", "PEN"),
)
_VALORES_MONEDA = {
    "PEN": "PEN",
    "S/": "PEN",
    "S/.": "PEN",
    "SOLES": "PEN",
    "USD": "USD",
    "US$": "USD",
    "$": "USD",
    "DOLARES": "USD",
    "DÓLARES": "USD",
}
# Acepta "1234.50" y "1,234.50"; rechaza el formato europeo "1.234,50".
_MONTO_RE = re.compile(r"[-+]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?")
_FECHA_RE = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})")
_CODIGO_REF_RE = re.compile(r"\bREF[:.#\s]*\S+|\*\S*|#\S*|\b\d{4,}\b")


class IngestError(ValueError):
    """El archivo no se puede usar; el mensaje es apto para mostrarse al usuario."""


@dataclass
class IngestResult:
    df: pd.DataFrame
    avisos: list[str] = field(default_factory=list)


def cargar_csv(origen: str | Path | bytes | BinaryIO) -> IngestResult:
    """Lee un CSV de movimientos y lo devuelve normalizado, con avisos para el usuario."""
    crudo = _leer_crudo(origen)
    _validar_columnas(crudo)
    avisos: list[str] = []

    vacias = crudo[(crudo == "").all(axis=1)].index
    crudo = crudo.drop(index=vacias)
    _avisar_lineas(avisos, "Se eliminaron {n} filas vacías", vacias)

    fechas = _parsear_fechas(crudo["fecha"])
    montos, monedas = _parsear_montos_y_monedas(crudo, avisos)

    sin_fecha = crudo.index[fechas.isna()]
    sin_monto = crudo.index[montos.isna() & fechas.notna()]
    _avisar_lineas(avisos, "Se eliminaron {n} filas sin fecha válida", sin_fecha)
    _avisar_lineas(avisos, "Se eliminaron {n} filas sin monto válido", sin_monto)

    validas = fechas.notna() & montos.notna()
    if not validas.any():
        raise IngestError("El archivo no tiene ninguna transacción con fecha y monto válidos.")

    originales = crudo.loc[validas, "descripcion"].str.strip()
    _avisar_lineas(
        avisos,
        f"{{n}} filas no tienen descripción; se marcaron como '{DESCRIPCION_VACIA}'",
        originales.index[originales == ""],
    )
    df = pd.DataFrame(
        {
            "fecha": fechas[validas],
            "descripcion_original": originales,
            "descripcion": originales.map(limpiar_descripcion),
            "monto": montos[validas],
            "moneda": monedas[validas],
        }
    )
    return IngestResult(df=_finalizar(df), avisos=avisos)


def limpiar_descripcion(texto: str) -> str:
    """Normaliza una descripción: mayúsculas, sin códigos de referencia ni espacios extra."""
    limpio = _CODIGO_REF_RE.sub(" ", texto.upper())
    limpio = re.sub(r"\s+", " ", limpio).strip()
    return limpio or DESCRIPCION_VACIA


def _leer_crudo(origen: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    if isinstance(origen, bytes):
        datos = origen
    elif isinstance(origen, str | Path):
        datos = Path(origen).read_bytes()
    else:
        datos = origen.read()

    texto = _decodificar(datos)
    if not texto.strip():
        raise IngestError("El archivo está vacío.")

    cabecera = texto.splitlines()[0]
    separador = ";" if cabecera.count(";") > cabecera.count(",") else ","
    # skip_blank_lines=False mantiene la relación índice → línea del archivo.
    error_formato = IngestError(
        "El CSV está mal formado: alguna fila tiene más columnas que la cabecera. "
        "Revisa que los montos con coma estén entre comillas."
    )
    try:
        crudo = pd.read_csv(
            io.StringIO(texto),
            sep=separador,
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=False,
        ).fillna("")
    except pd.errors.ParserError as error:
        raise error_formato from error
    # Con exactamente una columna de más, pandas no falla: usa la primera como índice.
    if not isinstance(crudo.index, pd.RangeIndex):
        raise error_formato
    crudo.columns = [str(c).strip().lower() for c in crudo.columns]
    return crudo


def _decodificar(datos: bytes) -> str:
    try:
        return datos.decode("utf-8-sig")
    except UnicodeDecodeError:
        return datos.decode("latin-1")


def _validar_columnas(crudo: pd.DataFrame) -> None:
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in crudo.columns]
    if faltantes:
        raise IngestError(
            f"Faltan columnas obligatorias: {', '.join(faltantes)}. "
            f"El archivo debe tener: {', '.join(COLUMNAS_REQUERIDAS)} (y opcionalmente moneda)."
        )


def _parsear_fechas(columna: pd.Series) -> pd.Series:
    """Detecta dd/mm o mm/dd mirando todo el archivo; si es ambiguo, asume dd/mm (Perú)."""
    partes = columna.str.strip().str.fullmatch(_FECHA_RE.pattern)
    piezas = columna.str.strip().str.extract(_FECHA_RE)
    piezas = piezas[partes].astype(int)

    primero_mayor = (piezas[0] > 12).any()
    segundo_mayor = (piezas[1] > 12).any()
    if primero_mayor and segundo_mayor:
        raise IngestError(
            "Las fechas mezclan formatos dd/mm/aaaa y mm/dd/aaaa. "
            "Usa un solo formato en el archivo."
        )
    dia, mes = (1, 0) if segundo_mayor else (0, 1)

    fechas = pd.Series(pd.NaT, index=columna.index, dtype="datetime64[ns]")
    for idx, fila in piezas.iterrows():
        anio = fila[2] + 2000 if fila[2] < 100 else fila[2]
        try:
            fechas.loc[idx] = pd.Timestamp(year=anio, month=fila[mes], day=fila[dia])
        except ValueError:
            pass  # p. ej. 31/02: queda como NaT y la fila se elimina con aviso
    return fechas


def _parsear_montos_y_monedas(
    crudo: pd.DataFrame, avisos: list[str]
) -> tuple[pd.Series, pd.Series]:
    """El símbolo del monto tiene prioridad sobre la columna moneda; por defecto, PEN."""
    parseados = crudo["monto"].map(_parsear_monto)
    montos = parseados.map(lambda p: p[0]).astype(float)
    simbolos = parseados.map(lambda p: p[1])

    if "moneda" in crudo.columns:
        texto = crudo["moneda"].str.strip().str.upper()
        columna = texto.map(_VALORES_MONEDA)
        _avisar_lineas(
            avisos,
            "{n} filas tienen un valor de moneda no reconocido; se trataron como sin moneda",
            crudo.index[(texto != "") & columna.isna()],
        )
        _avisar_lineas(
            avisos,
            "En {n} filas el símbolo del monto no coincide con la columna moneda; "
            "se usó el símbolo",
            crudo.index[simbolos.notna() & columna.notna() & (simbolos != columna)],
        )
        _avisar_lineas(
            avisos,
            f"{{n}} filas no indican moneda; se asumió {MONEDA_POR_DEFECTO}",
            crudo.index[simbolos.isna() & columna.isna() & montos.notna()],
        )
    else:
        columna = pd.Series(None, index=crudo.index, dtype=object)
        avisos.append(
            f"El archivo no tiene columna 'moneda': se asumió {MONEDA_POR_DEFECTO} "
            "salvo en los montos con símbolo."
        )

    monedas = simbolos.fillna(columna).fillna(MONEDA_POR_DEFECTO)
    return montos, monedas


def _parsear_monto(texto: str) -> tuple[float | None, str | None]:
    limpio = texto.strip().upper()
    moneda = None
    for simbolo, codigo in _SIMBOLOS_MONEDA:
        if simbolo in limpio:
            moneda = codigo
            limpio = limpio.replace(simbolo, "")
            break
    limpio = limpio.replace(" ", "")
    if not _MONTO_RE.fullmatch(limpio):
        return None, moneda
    return float(limpio.replace(",", "")), moneda


def _avisar_lineas(avisos: list[str], plantilla: str, indices: pd.Index) -> None:
    """Agrega un aviso con las líneas del archivo afectadas (índice + 2 por la cabecera)."""
    if len(indices) == 0:
        return
    lineas = [str(i + 2) for i in indices[:MAX_LINEAS_EN_AVISO]]
    sufijo = ", …" if len(indices) > MAX_LINEAS_EN_AVISO else ""
    avisos.append(f"{plantilla.format(n=len(indices))} (líneas {', '.join(lineas)}{sufijo}).")


def _finalizar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("fecha", kind="stable").reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    df.insert(2, "mes", df["fecha"].dt.strftime("%Y-%m"))
    return df
