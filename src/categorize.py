"""Categorización de transacciones en capas: reglas → caché → LLM.

Las columnas que agrega cada capa son:
    categoria   str | None    valor de `Categoria`; None = pendiente para la siguiente capa
    capa        str | None    "reglas", "cache" o "llm"
    confianza   float | None  1.0 para reglas; la que reportó el LLM en cache/llm

La caché es por sesión (nunca compartida entre usuarios) y la clave es la descripción
ya normalizada por `ingest`.
"""

from __future__ import annotations

import re

import pandas as pd

from src.ingest import DESCRIPCION_VACIA
from src.schemas import Categoria

C = Categoria

# Se acepta la categoría del LLM solo si confianza > umbral; si no, va a Otros y no se cachea.
UMBRAL_CONFIANZA = 0.5

ResultadoLLM = tuple[Categoria, float]

# Cada frase debe aparecer como palabras completas y consecutivas en la descripción.
# Ante varias coincidencias gana la frase más larga (más específica); si empatan, la primera.
REGLAS: list[tuple[str, Categoria]] = [
    # Suscripciones (incluye variantes que chocan con comercios de otra categoría)
    ("RAPPI PRO", C.SUSCRIPCIONES),
    ("UBER ONE", C.SUSCRIPCIONES),
    ("NETFLIX", C.SUSCRIPCIONES),
    ("SPOTIFY", C.SUSCRIPCIONES),
    ("CHATGPT", C.SUSCRIPCIONES),
    ("OPENAI", C.SUSCRIPCIONES),
    ("DISNEY", C.SUSCRIPCIONES),
    ("PRIME VIDEO", C.SUSCRIPCIONES),
    ("AMAZON PRIME", C.SUSCRIPCIONES),
    ("HBO", C.SUSCRIPCIONES),
    ("YOUTUBE PREMIUM", C.SUSCRIPCIONES),
    ("CRUNCHYROLL", C.SUSCRIPCIONES),
    ("ICLOUD", C.SUSCRIPCIONES),
    ("GOOGLE ONE", C.SUSCRIPCIONES),
    # Comida y delivery
    ("RAPPI", C.COMIDA),
    ("PEDIDOSYA", C.COMIDA),
    ("PEDIDOS YA", C.COMIDA),
    ("UBER EATS", C.COMIDA),
    ("DIDI FOOD", C.COMIDA),
    ("KFC", C.COMIDA),
    ("BEMBOS", C.COMIDA),
    ("PIZZA HUT", C.COMIDA),
    ("PAPA JOHNS", C.COMIDA),
    ("STARBUCKS", C.COMIDA),
    ("MCDONALDS", C.COMIDA),
    ("NORKYS", C.COMIDA),
    # Supermercado
    ("WONG", C.SUPERMERCADO),
    ("PLAZA VEA", C.SUPERMERCADO),
    ("PLAZAVEA", C.SUPERMERCADO),
    ("TOTTUS", C.SUPERMERCADO),
    ("METRO", C.SUPERMERCADO),
    ("VIVANDA", C.SUPERMERCADO),
    ("MAKRO", C.SUPERMERCADO),
    ("MASS", C.SUPERMERCADO),
    ("TAMBO", C.SUPERMERCADO),
    ("OXXO", C.SUPERMERCADO),
    # Transporte
    ("UBER", C.TRANSPORTE),
    ("CABIFY", C.TRANSPORTE),
    ("INDRIVE", C.TRANSPORTE),
    ("DIDI", C.TRANSPORTE),
    ("METROPOLITANO", C.TRANSPORTE),
    ("PRIMAX", C.TRANSPORTE),
    ("REPSOL", C.TRANSPORTE),
    # Entretenimiento
    ("CINEPLANET", C.ENTRETENIMIENTO),
    ("CINEMARK", C.ENTRETENIMIENTO),
    ("CINEPOLIS", C.ENTRETENIMIENTO),
    ("TELETICKET", C.ENTRETENIMIENTO),
    ("JOINNUS", C.ENTRETENIMIENTO),
    ("TICKETMASTER", C.ENTRETENIMIENTO),
    ("STEAM", C.ENTRETENIMIENTO),
    ("PLAYSTATION", C.ENTRETENIMIENTO),
    # Salud
    ("INKAFARMA", C.SALUD),
    ("MIFARMA", C.SALUD),
    ("BOTICA", C.SALUD),
    ("BOTICAS", C.SALUD),
    ("FARMACIA", C.SALUD),
    ("CLINICA", C.SALUD),
    # Educación
    ("PENSION", C.EDUCACION),
    ("UNIVERSIDAD", C.EDUCACION),
    ("UPC", C.EDUCACION),
    ("PUCP", C.EDUCACION),
    ("UTEC", C.EDUCACION),
    ("COURSERA", C.EDUCACION),
    ("UDEMY", C.EDUCACION),
    ("PLATZI", C.EDUCACION),
    # Servicios (luz, agua, internet, celular)
    ("MOVISTAR", C.SERVICIOS),
    ("CLARO", C.SERVICIOS),
    ("ENTEL", C.SERVICIOS),
    ("BITEL", C.SERVICIOS),
    ("SEDAPAL", C.SERVICIOS),
    ("LUZ DEL SUR", C.SERVICIOS),
    ("ENEL", C.SERVICIOS),
    ("CALIDDA", C.SERVICIOS),
    # Transferencias (en ambas direcciones; el signo del monto dice si entra o sale)
    ("YAPE", C.TRANSFERENCIAS),
    ("PLIN", C.TRANSFERENCIAS),
    ("TUNKI", C.TRANSFERENCIAS),
    ("TRANSFERENCIA", C.TRANSFERENCIAS),
    ("TRANSF", C.TRANSFERENCIAS),
    # Ingresos
    ("PLANILLA", C.INGRESOS),
    ("HABERES", C.INGRESOS),
    ("SUELDO", C.INGRESOS),
    ("REMUNERACION", C.INGRESOS),
]

_SEPARADORES_RE = re.compile(r"[^A-Z0-9ÁÉÍÓÚÑÜ]+")


def tokenizar(texto: str) -> list[str]:
    """Separa en palabras por espacios, puntos, guiones y cualquier otro signo."""
    return [t for t in _SEPARADORES_RE.split(texto.upper()) if t]


# sorted es estable: entre frases del mismo largo se mantiene el orden de REGLAS.
_REGLAS_POR_ESPECIFICIDAD = sorted(
    ((tuple(tokenizar(frase)), categoria) for frase, categoria in REGLAS),
    key=lambda regla: len(regla[0]),
    reverse=True,
)


def aplicar_reglas(descripcion: str) -> Categoria | None:
    """Devuelve la categoría de la regla más específica que coincide, o None."""
    if descripcion == DESCRIPCION_VACIA:
        return C.OTROS
    palabras = tokenizar(descripcion)
    for frase, categoria in _REGLAS_POR_ESPECIFICIDAD:
        if _contiene_frase(palabras, frase):
            return categoria
    return None


def categorizar_por_reglas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega categoria/capa/confianza donde resuelve una regla; el resto queda en None."""
    resultado = df.copy()
    # Categoria es un StrEnum, así que la columna queda con el texto de cada categoría.
    categorias = resultado["descripcion"].map(aplicar_reglas)
    resueltas = categorias.notna()
    resultado["categoria"] = categorias
    resultado["capa"] = resueltas.map({True: "reglas", False: None})
    resultado["confianza"] = resueltas.map({True: 1.0, False: None}).astype(float)
    return resultado


def _contiene_frase(palabras: list[str], frase: tuple[str, ...]) -> bool:
    largo = len(frase)
    return any(tuple(palabras[i : i + largo]) == frase for i in range(len(palabras) - largo + 1))


class CacheCategorias:
    """Caché de una sesión: descripción normalizada → (categoría, confianza)."""

    def __init__(self) -> None:
        self._datos: dict[str, ResultadoLLM] = {}

    def __len__(self) -> int:
        return len(self._datos)

    def buscar(self, descripcion: str) -> ResultadoLLM | None:
        return self._datos.get(descripcion)

    def guardar(self, descripcion: str, categoria: Categoria, confianza: float) -> None:
        """Solo guarda resultados confiables, para no propagar un error a futuras filas."""
        if confianza > UMBRAL_CONFIANZA:
            self._datos[descripcion] = (categoria, confianza)


def categorizar_por_cache(df: pd.DataFrame, cache: CacheCategorias) -> pd.DataFrame:
    """Resuelve las filas pendientes cuya descripción ya está en la caché."""
    resultado = df.copy()
    pendientes = resultado.loc[resultado["categoria"].isna(), "descripcion"]
    aciertos = pendientes.map(cache.buscar).dropna()
    _asignar(resultado, aciertos.index, aciertos, capa="cache")
    return resultado


def pendientes_unicos(df: pd.DataFrame) -> list[str]:
    """Descripciones sin categoría, sin repetir: es lo único que se manda al LLM."""
    return df.loc[df["categoria"].isna(), "descripcion"].drop_duplicates().tolist()


def aplicar_resultados_llm(
    df: pd.DataFrame, resultados: dict[str, ResultadoLLM], cache: CacheCategorias
) -> pd.DataFrame:
    """Vuelca las respuestas del LLM: la primera aparición cuenta como "llm", el resto "cache".

    Con confianza <= UMBRAL_CONFIANZA la fila va a Otros (conservando su confianza) y no se
    guarda en la caché. Las descripciones sin respuesta quedan pendientes.
    """
    for descripcion, (categoria, confianza) in resultados.items():
        cache.guardar(descripcion, categoria, confianza)

    resultado = df.copy()
    pendientes = resultado.loc[resultado["categoria"].isna(), "descripcion"]
    pendientes = pendientes[pendientes.isin(list(resultados))]
    aceptados = pendientes.map(lambda d: _aceptar(*resultados[d]))

    primera_vez = ~pendientes.duplicated()
    _asignar(resultado, pendientes.index[primera_vez], aceptados[primera_vez], capa="llm")
    _asignar(resultado, pendientes.index[~primera_vez], aceptados[~primera_vez], capa="cache")
    return resultado


def _aceptar(categoria: Categoria, confianza: float) -> ResultadoLLM:
    return (categoria if confianza > UMBRAL_CONFIANZA else C.OTROS, confianza)


def _asignar(df: pd.DataFrame, indices: pd.Index, valores: pd.Series, capa: str) -> None:
    if len(indices) == 0:
        return
    df.loc[indices, "categoria"] = valores.map(lambda r: r[0]).astype(str)
    df.loc[indices, "confianza"] = valores.map(lambda r: r[1]).astype(float)
    df.loc[indices, "capa"] = capa
