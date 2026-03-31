import numpy as np
import pandas as pd

from pgmpy.estimators import MaximumLikelihoodEstimator as MLE
from pgmpy.factors.hybrid.FunctionalCPD_Refactor import FunctionalCPD
from pgmpy.models.FunctionalBayesianNetwork_Refactor import FunctionalBayesianNetwork as FunctionalBN


def test_fit_learns_mixed_tabular_and_linear_cpds():
    rng = np.random.default_rng(42)
    n_samples = 1000

    a_data = rng.integers(0, 2, size=n_samples)
    b_data = rng.integers(0, 2, size=n_samples)
    c_data = rng.normal(loc=5.0, scale=2.0, size=n_samples)
    d_data = 2.5 * a_data - 1.5 * b_data + 3.0 * c_data + rng.normal(loc=0.0, scale=1.0, size=n_samples)

    data = pd.DataFrame({"A": a_data, "B": b_data, "C": c_data, "D": d_data})

    model = FunctionalBN([("A", "D"), ("B", "D"), ("C", "D")])

    cpd_a = FunctionalCPD(variable="A", tag="tabular", estimator=MLE)
    cpd_b = FunctionalCPD(variable="B", tag="tabular", estimator=MLE)
    cpd_c = FunctionalCPD(variable="C", tag="linear", estimator="MLE")
    cpd_d = FunctionalCPD(variable="D", tag="linear", estimator="MLE")

    model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d)
    model.fit(data)

    fitted_a = model.get_cpds("A").fitted_cpd_
    fitted_b = model.get_cpds("B").fitted_cpd_
    fitted_c = model.get_cpds("C").fitted_cpd_
    fitted_d = model.get_cpds("D").fitted_cpd_

    np.testing.assert_allclose(fitted_a.get_values().reshape(-1), [0.5, 0.5], atol=0.08)
    np.testing.assert_allclose(fitted_b.get_values().reshape(-1), [0.5, 0.5], atol=0.08)

    np.testing.assert_allclose(fitted_c.beta, [5.0], atol=0.2)
    np.testing.assert_allclose(fitted_c.std, 2.0, atol=0.2)

    np.testing.assert_allclose(fitted_d.beta[0], 0.0, atol=0.25)
    np.testing.assert_allclose(fitted_d.beta[1:], [2.5, -1.5, 3.0], atol=0.2)
    np.testing.assert_allclose(fitted_d.std, 1.0, atol=0.15)
