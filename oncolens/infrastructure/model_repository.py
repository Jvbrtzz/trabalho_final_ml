"""Acesso aos artefatos persistidos com joblib."""
from pathlib import Path
from typing import Any

import joblib


class ModelArtifactError(RuntimeError):
    """Indica artefatos ausentes ou incompatíveis."""


class JoblibModelRepository:
    """Carrega o pipeline e seus metadados sem conhecer regras de negócio."""

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self._model_path = model_path
        self._metadata_path = metadata_path

    def load(self) -> tuple[Any, dict]:
        missing = [
            path for path in (self._model_path, self._metadata_path) if not path.exists()
        ]
        if missing:
            names = ", ".join(path.name for path in missing)
            raise ModelArtifactError(
                f"Artefato(s) não encontrado(s): {names}. Execute o notebook_ml.ipynb."
            )

        try:
            model = joblib.load(self._model_path)
            metadata = joblib.load(self._metadata_path)
        except Exception as error:
            raise ModelArtifactError("Não foi possível carregar os artefatos do modelo.") from error

        required_keys = {"api_features", "model_features", "threshold"}
        if not isinstance(metadata, dict) or not required_keys.issubset(metadata):
            raise ModelArtifactError("O arquivo de metadados possui formato inválido.")
        if len(metadata["api_features"]) != len(metadata["model_features"]):
            raise ModelArtifactError("As listas de atributos dos metadados são incompatíveis.")

        return model, metadata

