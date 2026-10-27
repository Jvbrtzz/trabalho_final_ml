"""Composição das dependências e criação da aplicação FastAPI."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from oncolens.api.routes.health import router as health_router
from oncolens.api.routes.prediction import create_prediction_router
from oncolens.config import Settings, settings as default_settings
from oncolens.infrastructure.model_repository import JoblibModelRepository
from oncolens.services.prediction_service import PredictionService
from oncolens.web.routes import create_web_router


def create_app(settings: Settings = default_settings) -> FastAPI:
    """Application factory: único local que conhece todas as implementações."""
    repository = JoblibModelRepository(settings.model_path, settings.metadata_path)
    model, metadata = repository.load()
    prediction_service = PredictionService(model, metadata)

    application = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
    )
    application.mount(
        "/static",
        StaticFiles(directory=settings.static_dir),
        name="static",
    )
    application.include_router(create_web_router(settings.static_dir))
    application.include_router(health_router, prefix="/api/v1")
    application.include_router(
        create_prediction_router(prediction_service),
        prefix="/api/v1",
    )
    return application

