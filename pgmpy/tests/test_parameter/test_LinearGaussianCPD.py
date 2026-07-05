import numpy as np
import pandas as pd
import pytest

from pgmpy.parameter.LinearGaussianCPD import LinearGaussianCPD


@pytest.fixture
def continue_data():
    """
    # P( Y | A, B, C ) = N( 1 + 2A + 5B + 7C, 3 )
    # MLE result

    # Case 1-1: root node with MLE case
    # Case 1-2: root node with Unbias case
    # Case 2-1: root node with MLE, sample_weight case
    # Case 2-1: root node with Unbias, sample_weight case
    # Case 3-1: not root node with MLE case
    # Case 3-2: not root node with Unbias case
    # Case 4-1: not root node with MLE, sample_weight case
    # Case 4-2: not root node with Unbias, sample_weight case
    # Case 5: root node with Bayesian estimate case
    # Case 6: root node with Bayesian estimate, sample_weight case
    # Case 7: not root node with Bayesian estimate case
    # Case 8: not root node with Bayesian estimate, sample_weight case

    """
    rng = np.random.default_rng(seed=42)
    n = 1000

    A = rng.normal(size=n)
    B = rng.normal(size=n)
    C = rng.normal(size=n)

    mean_y = 1 + 2 * A + 5 * B + 7 * C
    X = pd.DataFrame(
        {
            "A": A,
            "B": B,
            "C": C,
        }
    )
    y = pd.DataFrame(
        {
            "y": rng.normal(loc=mean_y, scale=3),
        }
    )
    return X, y


class TestLinearGaussianCPD:
    def test_base_parameter_default(self):
        parameter = LinearGaussianCPD()

        assert parameter.__class__.__name__ == "LinearGaussianCPD"
        assert parameter.get_class_tag("variable_type") == "continous"
        assert parameter.get_class_tag("produces_factor") is False
        assert parameter.get_class_tag("missing") is False
        assert parameter.get_class_tag("is_linear_gaussian") is True
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("python_dependencies") == ()

    def test_fit(self, continue_data):
        # Case 1-1: root node with MLE case
        _, y = continue_data
        parameter = LinearGaussianCPD()
        parameter.fit(y)

        assert np.isclose(y["y"].mean(), parameter.beta_)
        assert np.isclose(y["y"].std(ddof=0), parameter.std_)

        # Case 1-2: root node with Unbias case
        _, y = continue_data
        parameter = LinearGaussianCPD()
        parameter.fit(y)

        assert np.isclose(y["y"].mean(), parameter.beta_)
        assert np.isclose(y["y"].std(ddof=1), parameter.std_)

        # Case 2-1: root node with MLE, sample_weight case
        _, y = continue_data
        # sample_weight =
        parameter = LinearGaussianCPD()
        parameter.fit(y, sample_weight=sample_weight)

        s = y["y"].to_numpy()
        w = np.asarray(sample_weight, dtype=float)

        expected_mean = np.sum(w * s) / np.sum(w)
        expected_std = np.sqrt(np.sum(w * (s - expected_mean) ** 2) / np.sum(w))

        assert np.isclose(parameter.beta_[0], expected_mean)
        assert np.isclose(parameter.std_, expected_std)

        # Case 2-2: root node with Unbias, sample_weight case
        _, y = continue_data
        # sample_weight =
        parameter = LinearGaussianCPD()
        parameter.fit(y, sample_weight=sample_weight)

        # Case 3-1: not root node with MLE case
        X, y = continue_data
        parameter = LinearGaussianCPD()

        parameter.fit(X, y)

        # Case 3-2: not root node with Unbias case
        _, y = continue_data
        parameter = LinearGaussianCPD()

        parameter.fit(X, y)


        # Case 4-1: not root node with MLE, sample_weight case
        _, y = continue_data
        # sample_weight =
        parameter = LinearGaussianCPD()

        parameter.fit(X, y, sample_weight)

        # Case 4-2: not root node with Unbias, sample_weight case
        _, y = continue_data
        # sample_weight =
        parameter = LinearGaussianCPD()

        parameter.fit(X, y, sample_weight)

        # Case 5: root node with Bayesian estimate case
        # Case 6: root node with Bayesian estimate, sample_weight case
        # Case 7: not root node with Bayesian estimate case
        # Case 8: not root node with Bayesian estimate, sample_weight case

    def test_predict_proba(self, continue_data): ...

    def test_set_values(self): ...
