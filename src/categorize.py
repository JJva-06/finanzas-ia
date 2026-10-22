"""Categorización de transacciones en capas: reglas → caché → LLM.

Las columnas que agrega cada capa son:
    categoria   str | None    valor de `Categoria`; None = pendiente para la siguiente capa
    capa        str | None    "reglas", "cache", "llm" o "sin_llm" (el LLM no estaba disponible)
    confianza   float | None  1.0 para reglas; la que reportó el LLM en cache/llm

La caché es por sesión (nunca compartida entre usuarios) y la clave es la descripción
ya normalizada por `ingest`. Al LLM solo se le manda la descripción y si es cargo o abono:
nunca montos, fechas ni las filas que ya resolvieron las reglas.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from src.ingest import DESCRIPCION_VACIA
from src.llm import LLMNoDisponible, ProveedorLLM, Uso
from src.schemas import Categoria, RespuestaLote

C = Categoria

# Se acepta la categoría del LLM solo si confianza > umbral; si no, va a Otros.
UMBRAL_CONFIANZA = 0.5
TAMANO_LOTE = 40
# Tope de descripciones únicas por archivo: protege el saldo de la API en la demo pública.
MAX_DESCRIPCIONES_LLM = 100

PROMPT_SISTEMA = f"""Clasificas movimientos de estados de cuenta bancarios de personas en Perú.

Recibirás una lista JSON de movimientos. Cada uno tiene:
- id: identificador del movimiento en esta lista.
- descripcion: texto del banco, ya normalizado.
- tipo: "cargo" (sale dinero), "abono" (entra dinero) o "mixto" (aparece de ambas formas).

Asigna a cada id exactamente una de estas categorías:
- {C.COMIDA}: restaurantes, cafés, apps de delivery.
- {C.SUPERMERCADO}: supermercados, bodegas, tiendas de conveniencia.
- {C.TRANSPORTE}: taxis, apps de movilidad, transporte público, combustible.
- {C.SUSCRIPCIONES}: servicios de pago recurrente (streaming, software, membresías).
- {C.ENTRETENIMIENTO}: cine, eventos, videojuegos, ocio.
- {C.SALUD}: farmacias, clínicas, consultas, gimnasios.
- {C.EDUCACION}: universidades, cursos, librerías, útiles, copias e impresiones.
- {C.SERVICIOS}: luz, agua, gas, internet, telefonía.
- {C.TRANSFERENCIAS}: envíos o recepciones de dinero entre personas.
- {C.INGRESOS}: sueldo, honorarios y otros ingresos.
- {C.OTROS}: cualquier cosa que no encaje o que no se pueda determinar.

Reglas:
- Un cargo nunca es {C.INGRESOS}.
- Prefijos como IZI, MERPAGO, NIUBIZ, CULQI, VENDEMAS, PAYPAL o DLC son pasarelas de pago,
  no el comercio: clasifica según lo que viene después.
- No adivines: si la descripción no da pistas suficientes, usa {C.OTROS} con confianza baja.
- confianza va de 0 a 1: 0.9 o más si el comercio es inequívoco, entre 0.6 y 0.8 si es
  plausible pero ambiguo, 0.5 o menos si es una suposición.
- Devuelve exactamente un resultado por cada id recibido."""

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
        """Una respuesta dudosa se guarda como Otros: se recuerda el "no sé", no la adivinanza,
        y así no se vuelve a pagar por consultarla en la misma sesión."""
        self._datos[descripcion] = _aceptar(categoria, confianza)


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

    Con confianza <= UMBRAL_CONFIANZA la fila va a Otros (conservando su confianza), y así
    queda en la caché. Las descripciones sin respuesta quedan pendientes.
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


@dataclass
class ResumenCategorizacion:
    """Lo que la UI muestra como punto de venta: quién resolvió qué y cuánto costó."""

    por_capa: dict[str, int]
    descripciones_enviadas: int
    baja_confianza: int
    tokens_entrada: int
    tokens_salida: int
    costo_usd: float | None
    avisos: list[str] = field(default_factory=list)


def categorizar(
    df: pd.DataFrame, cache: CacheCategorias, llm: ProveedorLLM | None
) -> tuple[pd.DataFrame, ResumenCategorizacion]:
    """Pipeline completo: reglas → caché → LLM (solo descripciones únicas pendientes)."""
    resultado = categorizar_por_cache(categorizar_por_reglas(df), cache)
    pendientes = _items_para_llm(resultado)
    items, excedentes = pendientes[:MAX_DESCRIPCIONES_LLM], pendientes[MAX_DESCRIPCIONES_LLM:]
    avisos: list[str] = []
    entrada_antes, salida_antes = (llm.uso.tokens_entrada, llm.uso.tokens_salida) if llm else (0, 0)

    if items:
        try:
            if llm is None:
                raise LLMNoDisponible("La categorización con IA no está configurada.")
            respuestas = categorizar_con_llm(items, llm)
            resultado = aplicar_resultados_llm(resultado, respuestas, cache)
        except LLMNoDisponible as error:
            resultado = _marcar_sin_llm(resultado)
            avisos.append(f"{error} {len(items)} comercios desconocidos quedaron en 'Otros'.")
    if excedentes:
        resultado = _marcar_sin_llm(resultado)
        avisos.append(
            f"{len(excedentes)} comercios poco frecuentes no se categorizaron con IA "
            f"(máximo {MAX_DESCRIPCIONES_LLM} por archivo) y quedaron en 'Otros'."
        )

    tokens_entrada = llm.uso.tokens_entrada - entrada_antes if llm else 0
    tokens_salida = llm.uso.tokens_salida - salida_antes if llm else 0
    resumen = ResumenCategorizacion(
        por_capa=resultado["capa"].value_counts().to_dict(),
        descripciones_enviadas=len(items),
        baja_confianza=int((resultado["confianza"] <= UMBRAL_CONFIANZA).sum()),
        tokens_entrada=tokens_entrada,
        tokens_salida=tokens_salida,
        costo_usd=_costo(llm, tokens_entrada, tokens_salida),
        avisos=avisos,
    )
    return resultado, resumen


def categorizar_con_llm(items: list[tuple[str, str]], llm: ProveedorLLM) -> dict[str, ResultadoLLM]:
    """Categoriza (descripcion, tipo) en lotes. Lo que falla dos veces va a Otros con confianza 0.

    Solo se reintenta lo que no validó, una vez.
    """
    resultados: dict[str, ResultadoLLM] = {}
    for inicio in range(0, len(items), TAMANO_LOTE):
        lote = items[inicio : inicio + TAMANO_LOTE]
        respuestas = _consultar_lote(lote, llm)
        faltantes = [item for item in lote if item[0] not in respuestas]
        if faltantes:
            respuestas |= _consultar_lote(faltantes, llm)
        for descripcion, _ in lote:
            resultados[descripcion] = respuestas.get(descripcion, (C.OTROS, 0.0))
    return resultados


def _items_para_llm(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(descripcion, tipo) de cada descripción pendiente única, de la más a la menos frecuente.

    El orden importa por el tope: las primeras son las que más movimientos categorizan.
    """
    pendientes = df.loc[df["categoria"].isna(), ["descripcion", "monto"]]
    grupos = pendientes.groupby("descripcion", sort=False)["monto"]
    resumen = pd.DataFrame({"tipo": grupos.agg(_tipo_por_signo), "veces": grupos.size()})
    resumen = resumen.sort_values("veces", ascending=False, kind="stable")
    return list(resumen["tipo"].items())


def _tipo_por_signo(montos: pd.Series) -> str:
    if (montos > 0).all():
        return "abono"
    if (montos <= 0).all():
        return "cargo"
    return "mixto"


def _consultar_lote(lote: list[tuple[str, str]], llm: ProveedorLLM) -> dict[str, ResultadoLLM]:
    """Una llamada al LLM. Devuelve solo los resultados válidos, indexados por descripción."""
    contenido = json.dumps(
        [{"id": i, "descripcion": d, "tipo": t} for i, (d, t) in enumerate(lote)],
        ensure_ascii=False,
    )
    respuesta = llm.parse(PROMPT_SISTEMA, contenido, RespuestaLote)
    if respuesta is None:
        return {}

    ids = [r.id for r in respuesta.resultados]
    validos: dict[str, ResultadoLLM] = {}
    for r in respuesta.resultados:
        # Pydantic garantiza la forma; esto descarta lo que el esquema no puede garantizar.
        if 0 <= r.id < len(lote) and ids.count(r.id) == 1 and 0 <= r.confianza <= 1:
            validos[lote[r.id][0]] = (r.categoria, r.confianza)
    return validos


def _marcar_sin_llm(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()
    pendientes = resultado["categoria"].isna()
    resultado.loc[pendientes, "categoria"] = C.OTROS.value
    resultado.loc[pendientes, "capa"] = "sin_llm"
    return resultado


def _costo(llm: ProveedorLLM | None, tokens_entrada: int, tokens_salida: int) -> float | None:
    if llm is None:
        return 0.0
    return Uso(tokens_entrada=tokens_entrada, tokens_salida=tokens_salida).costo_usd(llm.modelo)
