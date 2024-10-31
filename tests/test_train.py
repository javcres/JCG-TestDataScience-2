import os

from jcg_testdatascience_2.config.core import TRAINED_MODEL_DIR, config
from jcg_testdatascience_2.train_pipeline import run_training


def test_model_training():  # WARNING this test can be costly to execute
    run_training()

    files = os.listdir(TRAINED_MODEL_DIR)
    model_file_found = False
    scaler_file_found = False
    # Check the trained pipeline exists in the output directory
    for file in files:
        if file.startswith(config.app_config.pipeline_save_file):
            model_file_found = True

        if file.startswith(config.app_config.pipeline_scaler_save_file):
            scaler_file_found = True

    assert model_file_found and scaler_file_found
