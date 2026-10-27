"""Rota de predição e adaptação entre HTTP e domínio."""
from fastapi import APIRouter

from oncolens.schemas.prediction import PatientFeatures, PredictionResponse
from oncolens.services.prediction_service import PredictionService


DISCLAIMER = "Resultado educacional; não substitui diagnóstico médico."


def create_prediction_router(service: PredictionService) -> APIRouter:
    router = APIRouter(tags=["Predição"])

    @router.post("/predictions", response_model=PredictionResponse)
    def predict(features: PatientFeatures) -> PredictionResponse:
        result = service.predict(features.model_dump())
        return PredictionResponse(
            classe=result.predicted_class,
            classe_codigo=result.class_code,
            probabilidade_maligno=result.malignant_probability,
            probabilidade_classe_prevista=result.predicted_class_probability,
            modelo=result.model_name,
            aviso=DISCLAIMER,
        )

    return router

