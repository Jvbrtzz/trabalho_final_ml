"""Objetos de domínio relacionados à classificação."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    class_code: int
    malignant_probability: float
    predicted_class_probability: float
    model_name: str

