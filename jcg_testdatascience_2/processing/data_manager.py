from typing import List

import joblib
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RandomForest

from jcg_testdatascience_2 import __version__ as _version
from jcg_testdatascience_2.config.core import (DATASET_DIR, TRAINED_MODEL_DIR,
                                               config)


def load_dataset() -> TimeSeries:
    """
    Loads the trainig data.
    """
    series = TimeSeries.from_csv(
        DATASET_DIR / config.app_config.training_data,
        config.pipeline_config.date,
        [config.pipeline_config.target] + config.pipeline_config.past_covariates,
    )

    return series


def save_pipeline(
    *, pipeline_base_name, pipeline_to_persist: RandomForest | Scaler
) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models.
    """

    # Prepare versioned save file name
    save_file_name = f"{pipeline_base_name}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_delete=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> RandomForest | Scaler:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_delete: List[str]) -> None:
    """
    Remove old pipelines.
    """
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name in files_to_delete:
            model_file.unlink()
