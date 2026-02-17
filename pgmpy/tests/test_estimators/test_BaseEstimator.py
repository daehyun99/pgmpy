import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators import BaseEstimator


@pytest.fixture
def d1():
    data = pd.DataFrame(
        data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
    )
    yield data
    del data


@pytest.fixture
def d2():
    return pd.DataFrame(
        data={
            "A": [0, np.nan, 1],
            "B": [0, 1, 0],
            "C": [1, 1, np.nan],
            "D": [np.nan, "Y", np.nan],
        }
    )


@pytest.fixture
def titanic_data():
    return pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")


class TestBaseEstimator:

    def test_state_count(self, d1):
        e = BaseEstimator(d1)

        assert e.state_counts("A").values.tolist() == [[2], [1]]
        assert e.state_counts("C", ["A", "B"]).values.tolist() == [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]

    def test_missing_data(self, d2):
        e = BaseEstimator(d2, state_names={"C": [0, 1]})

        assert e.state_counts("A").values.tolist() == [[1], [1]]
        assert e.state_counts("C", parents=["A", "B"]).values.tolist() == [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ]
