from darts import TimeSeries

from jcg_testdatascience_2.config.core import config
from jcg_testdatascience_2.processing.data_manager import load_dataset


def test_load_data():
    data = load_dataset()

    assert isinstance(data, TimeSeries)
    assert data.shape == (2083, 7, 1)
    for var in config.pipeline_config.past_covariates:
        assert var in data.columns
