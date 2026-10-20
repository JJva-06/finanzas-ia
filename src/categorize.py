"""Categorización de transacciones en capas: reglas → caché → LLM.

Por ahora solo la capa de reglas. Las columnas que agrega cada capa son:
    categoria   str | None    valor de `Categoria`; None = pendiente para la siguiente capa
    capa        str | None    "reglas", "cache" o "llm"
    confianza   float | None  1.0 para reglas
"""

from __future__ import annotations

import re

import pandas as pd

from src.ingest import DESCRIPCION_VACIA
from src.schemas import Categoria

C = Categoria

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
