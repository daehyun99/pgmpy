import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators import MaximumLikelihoodEstimator as MLE
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.hybrid.FunctionalCPD_Refactor import FunctionalCPD
from pgmpy.models.FunctionalBayesianNetwork_Refactor import FunctionalBayesianNetwork


def test_fit_learns_tabular_and_linear_cpds_from_mixed_specifications():
    rng = np.random.default_rng(42)
    n_samples = 5000

    a_data = rng.integers(0, 2, size=n_samples)
    b_data = rng.integers(0, 2, size=n_samples)

    c_prob = np.where((a_data == 1) | (b_data == 1), 0.9, 0.1)
    c_data = rng.binomial(n=1, p=c_prob)

    data = pd.DataFrame({"A": a_data, "B": b_data, "C": c_data})

    model = FunctionalBayesianNetwork([("A", "C"), ("B", "C")])

    cpd_c = FunctionalCPD(variable="C", tag="tabular", estimator=MLE)
    cpd_a = FunctionalCPD(variable="A", tag="linear", estimator="MLE")
    model.add_cpds(cpd_c, cpd_a)

    model.fit(data)

    learned_cpd_c = model.get_cpds("C")
    learned_cpd_a = model.get_cpds("A")

    assert learned_cpd_c.is_fitted_ is True
    assert learned_cpd_a.is_fitted_ is True
    assert isinstance(learned_cpd_c.fitted_cpd_, TabularCPD)
    assert isinstance(learned_cpd_a.fitted_cpd_, LinearGaussianCPD)

    # P(C=1 | A=0, B=0) should be close to 0.1
    assert learned_cpd_c.fitted_cpd_.get_value(C=1, A=0, B=0) == pytest.approx(0.1, abs=0.04)

    # P(C=1 | A=1, B=0), P(C=1 | A=0, B=1), P(C=1 | A=1, B=1) should be close to 0.9
    assert learned_cpd_c.fitted_cpd_.get_value(C=1, A=1, B=0) == pytest.approx(0.9, abs=0.04)
    assert learned_cpd_c.fitted_cpd_.get_value(C=1, A=0, B=1) == pytest.approx(0.9, abs=0.04)
    assert learned_cpd_c.fitted_cpd_.get_value(C=1, A=1, B=1) == pytest.approx(0.9, abs=0.04)

    # A is Bernoulli(0.5), so intercept(mean)~0.5 and std~0.5 when modeled without parents.
    assert learned_cpd_a.fitted_cpd_.beta[0] == pytest.approx(0.5, abs=0.03)
    assert learned_cpd_a.fitted_cpd_.std == pytest.approx(0.5, abs=0.03)
