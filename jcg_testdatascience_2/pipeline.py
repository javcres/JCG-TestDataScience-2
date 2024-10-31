from darts.dataprocessing.transformers import Scaler
from darts.models import RandomForest

from jcg_testdatascience_2.config.core import config

# Scaler setup
electricity_transformer_scaler = Scaler()

# Model setup
electricity_transformer_model = RandomForest(
    lags=config.pipeline_config.final_model_lags,
    lags_past_covariates=config.pipeline_config.final_model_lags_past_covariates,
    output_chunk_length=config.pipeline_config.final_model_output_chunk_length,
    multi_models=config.pipeline_config.final_model_multi_models,
    random_state=config.pipeline_config.random_seed,
)
