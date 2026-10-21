"""Cliente del LLM: configuración, salida estructurada, errores y conteo de tokens.

La API key solo se lee de la variable de entorno ANTHROPIC_API_KEY, que viene de `.env`
en local o de los secrets de Streamlit Cloud. Nunca se escribe en el código.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import anthropic
import pydantic
from dotenv import load_dotenv

MODELO_POR_DEFECTO = "claude-haiku-5-5"
# USD por millón de tokens (entrada, salida), según la tabla de precios de Anthropic.
PRECIOS_USD_POR_MILLON = {"claude-haiku-5-5": (0.10, 0.50)}
MAX_TOKENS = 4096

T = TypeVar("T", bound=pydantic.BaseModel)


class LLMNoDisponible(Exception):
    """El LLM no se pudo usar; el mensaje es apto para mostrarse al usuario."""


@dataclass
class Uso:
    llamadas: int = 0
    tokens_entrada: int = 0
    tokens_salida: int = 0

    def costo_usd(self, modelo: str) -> float | None:
        precios = PRECIOS_USD_POR_MILLON.get(modelo)
        if precios is None:
            return None
        entrada, salida = precios
        return (self.tokens_entrada * entrada + self.tokens_salida * salida) / 1_000_000


class ProveedorLLM(Protocol):
    """Lo que `categorize` necesita de un LLM; los tests usan un doble que lo cumple."""

    modelo: str
    uso: Uso

    def parse(self, system: str, contenido: str, formato: type[T]) -> T | None: ...


class ProveedorChat(Protocol):
    """Lo que `chat` necesita de un LLM; los tests usan un doble que lo cumple."""

    modelo: str
    uso: Uso

    def conversar(
        self, system: str, messages: list[dict], tools: list[dict], tool_choice: dict | None = None
    ) -> Any: ...


class ClienteLLM:
    def __init__(self, cliente: Any | None = None, modelo: str | None = None) -> None:
        load_dotenv()
        self.modelo = modelo or os.getenv("LLM_MODEL") or MODELO_POR_DEFECTO
        self.uso = Uso()
        self._cliente = cliente

    def parse(self, system: str, contenido: str, formato: type[T]) -> T | None:
        """Devuelve la respuesta validada, o None si el modelo no produjo una salida válida.

        Los errores de la API (tras los reintentos automáticos del SDK) se convierten en
        LLMNoDisponible con un mensaje amigable.
        """
        try:
            respuesta = self._llamar(
                "parse",
                messages=[{"role": "user", "content": contenido}],
                system=system,
                output_format=formato,
            )
        except pydantic.ValidationError:
            return None
        if respuesta.stop_reason in ("refusal", "max_tokens"):
            return None
        return respuesta.parsed_output

    def conversar(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice: dict | None = None,
    ) -> Any:
        """Una vuelta de conversación con herramientas. Devuelve el Message del SDK."""
        extra = {"tool_choice": tool_choice} if tool_choice else {}
        return self._llamar("create", messages=messages, system=system, tools=tools, **extra)

    def _llamar(self, metodo: str, **kwargs: Any) -> Any:
        """Llama a messages.<metodo>, traduce los errores de la API y cuenta los tokens."""
        try:
            respuesta = getattr(self._obtener_cliente().messages, metodo)(
                model=self.modelo,
                max_tokens=MAX_TOKENS,
                output_config={"effort": "low"},
                **kwargs,
            )
        except anthropic.AuthenticationError as error:
            raise LLMNoDisponible("La API key de Anthropic no es válida.") from error
        except anthropic.RateLimitError as error:
            raise LLMNoDisponible("El servicio de IA está saturado; intenta en un rato.") from error
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as error:
            raise LLMNoDisponible("No se pudo conectar con el servicio de IA.") from error
        except anthropic.APIStatusError as error:
            raise LLMNoDisponible("El servicio de IA respondió con un error.") from error

        self.uso.llamadas += 1
        self.uso.tokens_entrada += respuesta.usage.input_tokens
        self.uso.tokens_salida += respuesta.usage.output_tokens
        return respuesta

    def _obtener_cliente(self) -> Any:
        if self._cliente is None:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise LLMNoDisponible("No hay API key de Anthropic configurada.")
            # El SDK reintenta solo 429, 5xx, timeouts y errores de conexión.
            self._cliente = anthropic.Anthropic(max_retries=2, timeout=60.0)
        return self._cliente
