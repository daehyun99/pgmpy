import numpy as np
import pandas as pd
import pytest
from skpro.distributions import Empirical

from pgmpy.base._base import _CoreGraph
from pgmpy.parameter import TabularCPD
from pgmpy.refactored_inference.LW import LikelihoodWeighting


def _binary_chain():
    model = _CoreGraph(edge_list=[("A", "B", "->")])
    model.SUPPORTED_EDGE_TYPES = frozenset(["->", "<-"])

    a = TabularCPD(categories={"A": [0, 1]}).fit(pd.DataFrame({"A": [0, 0, 1, 1]}))
    b = TabularCPD(categories={"B": [0, 1]}).fit(pd.DataFrame({"A": [0, 0, 1, 1]}), pd.DataFrame({"B": [0, 0, 0, 1]}))
    model.add_cpd("A", a)
    model.add_cpd("B", b)
    return model


def test_query_returns_empirical_distribution_for_evidence():
    distribution = LikelihoodWeighting().query(
        _binary_chain(), variables=["A"], evidence={"B": 1}, n_samples=2_000, seed=42
    )

    assert isinstance(distribution, Empirical)
    assert list(distribution.spl.columns) == ["A"]
    assert distribution.weights.sum() == pytest.approx(1.0)
    assert np.average(distribution.spl["A"].astype(float), weights=distribution.weights) == pytest.approx(1.0)


def test_query_applies_interventions_without_likelihood_weighting_them():
    distribution = LikelihoodWeighting().query(_binary_chain(), variables=["B"], do={"A": 0}, n_samples=2_000, seed=42)

    assert np.average(distribution.spl["B"].astype(float), weights=distribution.weights) == pytest.approx(0.0, abs=0.05)
