"""Configuração centralizada e independente do framework web."""
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    app_name: str = "OncoLens API"
    app_version: str = "2.0.0"
    app_description: str = (
        "API educacional de classificação com Regressão Logística."
    )
    model_path: Path = PROJECT_ROOT / "models" / "modelo_regressao_logistica.joblib"
    metadata_path: Path = PROJECT_ROOT / "models" / "metadados.joblib"
    static_dir: Path = PROJECT_ROOT / "static"


settings = Settings()

