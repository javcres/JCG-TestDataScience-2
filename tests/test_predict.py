from jcg_testdatascience_2.predict import make_prediction


def test_make_prediction():

    result = make_prediction()
    keys = result.keys()

    assert isinstance(result, dict)
    assert "predictions" in keys
    assert "version" in keys
    assert len(result["predictions"]) == 100
