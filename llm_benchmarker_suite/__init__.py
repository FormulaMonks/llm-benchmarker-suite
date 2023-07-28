# llm_benchmarking_suite/__init__.py
from .opencompass import *

# Specify which symbols to export from opencompass
__all__ = dir(opencompass)
