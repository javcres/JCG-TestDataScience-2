from jcg_testdatascience_2.config.core import config
from jcg_testdatascience_2.pipeline import (electricity_transformer_model,
                                            electricity_transformer_scaler)
from jcg_testdatascience_2.processing.data_manager import (load_dataset,
                                                           save_pipeline)


def run_training() -> None:
    """Train the model."""
    series = load_dataset()

    scaled_series = electricity_transformer_scaler.fit_transform(series)

    electricity_transformer_model.fit(
        series=scaled_series[config.pipeline_config.target],
        past_covariates=scaled_series[config.pipeline_config.past_covariates],
    )

    # Persist the trained scaler
    save_pipeline(
        pipeline_base_name=config.app_config.pipeline_scaler_save_file,
        pipeline_to_persist=electricity_transformer_scaler,
    )

    # Persist the trained model
    save_pipeline(
        pipeline_base_name=config.app_config.pipeline_save_file,
        pipeline_to_persist=electricity_transformer_model,
    )


if __name__ == "__main__":
    run_training()
