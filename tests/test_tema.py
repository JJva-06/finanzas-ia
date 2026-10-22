from src import tema


def test_notas_escapan_el_texto_del_llm() -> None:
    html = tema.notas(['<script>alert("x")</script> gastaste S/ 10.00'])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_lineas_kpi_escapan_y_marcan_el_total() -> None:
    html = tema.lineas_kpi(
        [("Ingreso <b>", "S/ 1.00")], ("Balance", "S/ 0.50"), sello="Gasto +5.0% vs. <i>"
    )
    assert "<b>" not in html and "<i>" not in html
    assert 'class="linea total"' in html
    assert 'class="sello"' in html


def test_sin_variacion_no_hay_sello() -> None:
    assert 'class="sello"' not in tema.lineas_kpi([], ("Balance", "S/ 0.00"), sello=None)


def test_hoja_de_estilos_existe_y_define_la_paleta() -> None:
    css = tema.hoja_de_estilos()
    assert css.startswith("<style>")
    for variable in ("--papel", "--tinta", "--sello"):
        assert variable in css
