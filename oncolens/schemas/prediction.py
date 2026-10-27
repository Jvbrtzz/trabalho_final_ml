"""Schemas Pydantic da API de predição."""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """Oito medidas médias extraídas da imagem digitalizada do núcleo celular."""

    model_config = ConfigDict(extra="forbid")

    mean_radius: Annotated[float, Field(ge=6, le=30, description="Raio médio")]
    mean_texture: Annotated[float, Field(ge=5, le=45, description="Textura média")]
    mean_perimeter: Annotated[float, Field(ge=40, le=200, description="Perímetro médio")]
    mean_area: Annotated[float, Field(ge=100, le=3000, description="Área média")]
    mean_smoothness: Annotated[
        float, Field(ge=0.04, le=0.18, description="Suavidade média")
    ]
    mean_compactness: Annotated[
        float, Field(ge=0.01, le=0.40, description="Compacidade média")
    ]
    mean_concavity: Annotated[
        float, Field(ge=0, le=0.50, description="Concavidade média")
    ]
    mean_concave_points: Annotated[
        float, Field(ge=0, le=0.25, description="Pontos côncavos médios")
    ]


class PredictionResponse(BaseModel):
    classe: str
    classe_codigo: int
    probabilidade_maligno: float
    probabilidade_classe_prevista: float
    modelo: str
    aviso: str


class HealthResponse(BaseModel):
    status: str
    modelo: str
    versao: str

