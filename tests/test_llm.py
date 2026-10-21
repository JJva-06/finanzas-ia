from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from src import llm as modulo_llm
from src.llm import ClienteLLM, LLMNoDisponible, Uso
from src.schemas import Categoria, RespuestaLote, ResultadoItem

RESPUESTA = RespuestaLote(
    resultados=[ResultadoItem(id=0, categoria=Categoria.COMIDA, confianza=0.9)]
)


class AnthropicFalso:
    """Imita client.messages.parse sin red: devuelve, o lanza, lo que se le indique."""

    def __init__(self, resultado: object) -> None:
        self.resultado = resultado
        self.kwargs: dict = {}
        self.messages = self

    def parse(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        if isinstance(self.resultado, Exception):
            raise self.resultado
        if callable(self.resultado):
            return self.resultado()
        return self.resultado


def _respuesta(parsed: object, stop_reason: str = "end_turn") -> SimpleNamespace:
    uso = SimpleNamespace(input_tokens=1000, output_tokens=200)
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason, usage=uso)


def test_parse_devuelve_salida_y_cuenta_tokens() -> None:
    falso = AnthropicFalso(_respuesta(RESPUESTA))
    cliente = ClienteLLM(cliente=falso, modelo="claude-haiku-5-5")

    assert cliente.parse("sistema", "contenido", RespuestaLote) == RESPUESTA
    assert cliente.uso == Uso(llamadas=1, tokens_entrada=1000, tokens_salida=200)
    assert falso.kwargs["output_format"] is RespuestaLote
    assert falso.kwargs["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_respuesta_incompleta_devuelve_none(stop_reason: str) -> None:
    cliente = ClienteLLM(cliente=AnthropicFalso(_respuesta(RESPUESTA, stop_reason)))
    assert cliente.parse("s", "c", RespuestaLote) is None


def test_salida_que_no_valida_devuelve_none() -> None:
    falso = AnthropicFalso(lambda: RespuestaLote.model_validate({"resultados": "x"}))
    assert ClienteLLM(cliente=falso).parse("s", "c", RespuestaLote) is None


def test_error_de_conexion_es_llm_no_disponible() -> None:
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x.test"))
    cliente = ClienteLLM(cliente=AnthropicFalso(error))
    with pytest.raises(LLMNoDisponible, match="conectar"):
        cliente.parse("s", "c", RespuestaLote)


def test_sin_api_key_es_llm_no_disponible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modulo_llm, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMNoDisponible, match="API key"):
        ClienteLLM().parse("s", "c", RespuestaLote)


def test_modelo_configurable_por_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modulo_llm, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    assert ClienteLLM().modelo == "claude-sonnet-5-5"


def test_costo_haiku() -> None:
    uso = Uso(tokens_entrada=1_000_000, tokens_salida=1_000_000)
    assert uso.costo_usd("claude-haiku-5-5") == pytest.approx(0.60)
    assert uso.costo_usd("modelo-desconocido") is None
