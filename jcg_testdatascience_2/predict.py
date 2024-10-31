import numpy as np
import pandas as pd
from darts import TimeSeries

from jcg_testdatascience_2 import __version__ as _version
from jcg_testdatascience_2.config.core import config
from jcg_testdatascience_2.processing.data_manager import load_pipeline

pipeline_scaler_file_name = (
    f"{config.app_config.pipeline_scaler_save_file}{_version}.pkl"
)
trained_electricity_transformer_scaler = load_pipeline(
    file_name=pipeline_scaler_file_name
)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
trained_electricity_transformer_model = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, n_periods: int = 100) -> dict:
    """Make a prediction using the saved pipeline to a certain number of periods."""
    if n_periods <= 0:
        raise ValueError("Number of periods must be a positive number")

    predictions = trained_electricity_transformer_model.predict(n_periods)

    # We need to add the same number of dummy series than variables to be able to unscale the data
    dates = pd.date_range(
        start=predictions.time_index[0],
        periods=n_periods,
        freq=config.pipeline_config.data_freq,
    )
    dummy_data = np.random.rand(100)
    dummy_series = TimeSeries.from_times_and_values(
        dates, dummy_data, columns=["Dummy Series"]
    )
    for _ in range(len(config.pipeline_config.past_covariates)):
        predictions = predictions.stack(dummy_series)

    unscaled_predictions = trained_electricity_transformer_scaler.inverse_transform(
        predictions
    )

    results = {
        "predictions": unscaled_predictions[config.pipeline_config.target]
        .data_array()
        .to_numpy()
        .flatten(),
        "version": _version,
    }

    return results
