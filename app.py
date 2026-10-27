"""Ponto de entrada ASGI mantido pequeno por design.

Execução: ``uvicorn app:app --reload``.
"""
from oncolens.application import create_app


app = create_app()

