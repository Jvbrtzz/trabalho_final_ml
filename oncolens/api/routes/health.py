"""Rota de verificação de disponibilidade."""
from fastapi import APIRouter

from oncolens import __version__
from oncolens.schemas.prediction import HealthResponse
from oncolens.services.prediction_service import PredictionService


router = APIRouter(tags=["Monitoramento"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        modelo=PredictionService.MODEL_NAME,
        versao=__version__,
    )

