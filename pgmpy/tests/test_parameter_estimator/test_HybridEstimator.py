import pandas as pd
import pytest

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models.Refactored_BayesianNetwork import BayesianNetwork
from pgmpy.parameter_estimator import HybridEstimator


class DummyRegressor:
    def __init__(self, variable):
        self.variable = variable
        self.fit_called = False

    def fit(self, X):
        self.fit_called = True
        self.seen_columns = list(X.columns)
        return self


def test_hybrid_estimator_fits_tabular_and_custom_models():
    model = BayesianNetwork(ebunch=[("diff", "grade"), ("grade", "score")])
    cpd_grade = TabularCPD(
        variable="grade",
        variable_card=2,
        values=[[0.5, 0.5], [0.5, 0.5]],
        evidence=["diff"],
        evidence_card=[2],
    )
    cpd_score = DummyRegressor(variable="score")
    model.cpds = [cpd_grade, cpd_score]

    data = pd.DataFrame({"diff": [0, 1, 0, 1], "grade": [1, 0, 1, 0], "score": [5.2, 3.4, 6.1, 2.9]})

    est = HybridEstimator()
    est.fit(model, data)

    assert len(est.parameters_) == 2
    assert est.parameters_[0].variable == "grade"
    assert est.parameters_[1].fit_called is True


def test_hybrid_estimator_raises_with_empty_cpds():
    model = BayesianNetwork(ebunch=[("a", "b")])
    data = pd.DataFrame({"a": [0, 1], "b": [1, 0]})

    with pytest.raises(ValueError, match="model.cpds is empty"):
        HybridEstimator().fit(model, data)
