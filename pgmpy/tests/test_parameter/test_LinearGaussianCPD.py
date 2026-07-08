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
        assert parameter.get_class_tag("variable_type") == "continuous"
        assert parameter.get_class_tag("produces_factor") is False
        assert parameter.get_class_tag("missing") is False
        assert parameter.get_class_tag("is_linear_gaussian") is True
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("python_dependencies") == ()

    def test_fit(self, continue_data):
        # Case 1-1: root node with MLE case
        _, y = continue_data
        parameter = LinearGaussianCPD(estimator="mle")
        parameter.fit(y)

        assert np.isclose(y["y"].mean(), parameter.beta_)
        assert np.isclose(y["y"].std(ddof=0), parameter.std_)

        # Case 1-2: root node with Unbias case
        _, y = continue_data
        parameter = LinearGaussianCPD(estimator="unbias")
        parameter.fit(y)

        assert np.isclose(y["y"].mean(), parameter.beta_)
        assert np.isclose(y["y"].std(ddof=1), parameter.std_)

        # Case 2-1: root node with MLE, sample_weight case
        _, y = continue_data
        sample_weight = np.zeros(len(y), dtype=float)
        sample_weight[:5] = 1.0
        parameter = LinearGaussianCPD(estimator="mle")
        parameter.fit(y, sample_weight=sample_weight)

        y_arr = y["y"].to_numpy(dtype=float)
        expected_mean = np.sum(sample_weight * y_arr) / np.sum(sample_weight)
        expected_std = np.sqrt(np.sum(sample_weight * (y_arr - expected_mean) ** 2) / np.sum(sample_weight))
        assert np.isclose(expected_mean, y_arr[:5].mean())
        assert np.isclose(expected_std, y_arr[:5].std(ddof=0))

        # Case 2-2: root node with Unbias, sample_weight case
        _, y = continue_data
        sample_weight = np.zeros(len(y), dtype=float)
        sample_weight[:5] = 1.0
        parameter = LinearGaussianCPD(estimator="unbias")
        parameter.fit(y, sample_weight=sample_weight)

        expected_mean = np.sum(sample_weight * y_arr) / np.sum(sample_weight)
        expected_std = np.sqrt(np.sum(sample_weight * (y_arr - expected_mean) ** 2) / (np.sum(sample_weight) - 1))

        assert np.isclose(np.ravel(parameter.beta_)[0], expected_mean)
        assert np.isclose(np.ravel(parameter.std_)[0], expected_std)

        # Case 3-1: not root node with MLE case
        X, y = continue_data
        from sklearn.linear_model import LinearRegression

        lm = LinearRegression().fit(X, y)
        y_pred = lm.predict(X)
        residuals = y.to_numpy() - y_pred
        ddof = 0  # MLE
        expected_beta = np.concatenate(
            [
                np.ravel(lm.intercept_),
                np.ravel(lm.coef_),
            ]
        )
        expected_std = np.sqrt(np.sum(residuals**2) / (len(residuals) - ddof))

        parameter = LinearGaussianCPD(estimator="mle")
        parameter.fit(X, y)

        assert np.allclose(parameter.beta_, expected_beta)
        assert np.isclose(parameter.std_, expected_std)

        # Case 3-2: not root node with Unbias case
        _, y = continue_data
        parameter = LinearGaussianCPD(estimator="unbias")

        parameter.fit(X, y)

        # Case 4-1: not root node with MLE, sample_weight case
        _, y = continue_data
        sample_weight = np.zeros(len(y), dtype=float)
        sample_weight[:5] = 1.0
        parameter = LinearGaussianCPD(estimator="mle")

        parameter.fit(X, y, sample_weight)

        expected_lm = LinearRegression().fit(
            X,
            y_arr,
            sample_weight=sample_weight,
        )

        expected_beta = np.concatenate(
            [
                np.ravel(expected_lm.intercept_),
                np.ravel(expected_lm.coef_),
            ]
        )

        residuals = y_arr - expected_lm.predict(X)
        expected_std = np.sqrt(np.sum(sample_weight * residuals**2) / np.sum(sample_weight))

        np.testing.assert_allclose(parameter.beta_, expected_beta)
        assert np.isclose(parameter.std_, expected_std)

        # Case 4-2: not root node with Unbias, sample_weight case
        X, y = continue_data
        y_arr = y.to_numpy(dtype=float).reshape(-1)
        sample_weight = np.zeros(len(y), dtype=float)
        sample_weight[:5] = 1.0
        parameter = LinearGaussianCPD(estimator="unbias")

        parameter.fit(X, y, sample_weight)

        expected_lm = LinearRegression().fit(
            X,
            y_arr,
            sample_weight=sample_weight,
        )

        expected_beta = np.concatenate(
            [
                np.ravel(expected_lm.intercept_),
                np.ravel(expected_lm.coef_),
            ]
        )

        residuals = y_arr - expected_lm.predict(X).reshape(-1)

        expected_std = np.sqrt(np.sum(sample_weight * residuals**2) / (np.sum(sample_weight) - (1 + X.shape[1])))

        np.testing.assert_allclose(parameter.beta_, expected_beta)
        assert np.isclose(parameter.std_, expected_std)

        # Case 5: root node with Bayesian estimate case
        # Case 6: root node with Bayesian estimate, sample_weight case
        # Case 7: not root node with Bayesian estimate case
        # Case 8: not root node with Bayesian estimate, sample_weight case

    def test_predict_proba(self, continue_data): ...

    def test_set_fitted_params(self):
        parameter = LinearGaussianCPD()

        assert hasattr(parameter, "beta_") is False
        beta = np.array([0, 1, 2])
        std = np.array(3)
        parameter.set_fitted_params(
            beta=beta,
            std=std,
            is_fitted=True,
        )
        assert hasattr(parameter, "beta_")
        assert hasattr(parameter, "std_")
        assert parameter.is_fitted is True
