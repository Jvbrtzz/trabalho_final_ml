"""Caso de uso de classificação de uma nova amostra."""
from collections.abc import Mapping
from typing import Any

import pandas as pd

from oncolens.domain.prediction import PredictionResult


class PredictionService:
    """Aplica o pipeline treinado sem depender de FastAPI ou Pydantic."""

    MODEL_NAME = "Regressão Logística"

    def __init__(self, model: Any, metadata: dict) -> None:
        self._model = model
        self._metadata = metadata

    def predict(self, features: Mapping[str, float]) -> PredictionResult:
        row = self._build_model_input(features)
        malignant_probability = float(self._model.predict_proba(row)[0, 1])
        threshold = float(self._metadata["threshold"])
        class_code = int(malignant_probability >= threshold)
        predicted_class = self._metadata.get("class_names", {}).get(
            class_code, "Maligno" if class_code else "Benigno"
        )
        predicted_probability = (
            malignant_probability if class_code else 1 - malignant_probability
        )

        return PredictionResult(
            predicted_class=predicted_class,
            class_code=class_code,
            malignant_probability=round(malignant_probability, 4),
            predicted_class_probability=round(predicted_probability, 4),
            model_name=self.MODEL_NAME,
        )

    def _build_model_input(self, features: Mapping[str, float]) -> pd.DataFrame:
        api_features = self._metadata["api_features"]
        missing = [name for name in api_features if name not in features]
        if missing:
            raise ValueError(f"Atributos ausentes: {', '.join(missing)}")

        values = [[features[name] for name in api_features]]
        return pd.DataFrame(values, columns=self._metadata["model_features"])

