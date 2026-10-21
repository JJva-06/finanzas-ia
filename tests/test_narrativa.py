import pytest

from src.insights import Hecho, redactar_con_plantilla
from src.llm import LLMNoDisponible, Uso
from src.narrativa import redactar_insights, usa_solo_numeros_del_hecho
from src.schemas import TextosInsights

AHORRO = Hecho(
    "ahorro",
    {
        "mes": "septiembre",
        "balance": "S/ 47.55",
        "mes_anterior": "agosto",
        "balance_anterior": "S/ 420.35",
    },
)
HORMIGA = Hecho(
    "hormiga", {"mes": "septiembre", "cantidad": "23", "umbral": "S/ 20.00", "total": "S/ 280.40"}
)


class LLMTextos:
    def __init__(self, resultado: object) -> None:
        self.modelo = "claude-haiku-5-5"
        self.uso = Uso()
        self.resultado = resultado
        self.contenido = ""

    def parse(self, system: str, contenido: str, formato: type) -> TextosInsights | None:
        self.contenido = contenido
        if isinstance(self.resultado, Exception):
            raise self.resultado
        return self.resultado


def test_usa_los_textos_del_llm_si_respetan_los_numeros() -> None:
    textos = [
        "Este mes ahorraste S/ 47.55, menos que los S/ 420.35 de agosto.",
        "Ojo: 23 compritas de menos de S/ 20.00 sumaron S/ 280.40.",
    ]
    llm = LLMTextos(TextosInsights(textos=textos))
    assert redactar_insights([AHORRO, HORMIGA], llm) == textos
    assert "S/ 47.55" in llm.contenido


def test_numero_inventado_cae_a_la_plantilla() -> None:
    textos = ["Ahorraste S/ 47.55, un 89% menos que en agosto.", "23 compras sumaron S/ 280.40."]
    resultado = redactar_insights([AHORRO, HORMIGA], LLMTextos(TextosInsights(textos=textos)))
    assert resultado == [redactar_con_plantilla(AHORRO), textos[1]]


@pytest.mark.parametrize(
    "llm",
    [
        None,
        LLMTextos(LLMNoDisponible("sin saldo")),
        LLMTextos(None),
        LLMTextos(TextosInsights(textos=["solo uno"])),
    ],
)
def test_sin_llm_o_respuesta_incompleta_usa_plantillas(llm: LLMTextos | None) -> None:
    plantillas = [redactar_con_plantilla(AHORRO), redactar_con_plantilla(HORMIGA)]
    assert redactar_insights([AHORRO, HORMIGA], llm) == plantillas


def test_control_de_numeros() -> None:
    assert usa_solo_numeros_del_hecho("Hiciste 23 compras por S/ 280.40.", HORMIGA)
    assert not usa_solo_numeros_del_hecho("Hiciste 24 compras.", HORMIGA)
    assert usa_solo_numeros_del_hecho("Sin números aquí.", HORMIGA)
