import anthropic
import pandas as pd
import plotly
import pydantic
import streamlit


def test_dependencias_importan() -> None:
    for modulo in (anthropic, pd, plotly, pydantic, streamlit):
        assert modulo.__version__


def test_python_version() -> None:
    import sys

    assert sys.version_info >= (3, 11)
