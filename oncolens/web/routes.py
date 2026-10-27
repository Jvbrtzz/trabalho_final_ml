"""Rota da página única da aplicação."""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


def create_web_router(static_dir: Path) -> APIRouter:
    router = APIRouter(include_in_schema=False)

    @router.get("/")
    def home() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return router

