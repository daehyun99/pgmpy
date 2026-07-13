import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from skpro.distributions.normal import Normal
from skpro.regression.gam import GAMRegressor

from pgmpy.parameter.adapter.skpro import SkproAdapter


def _conditional_mean(X):
    """
    Y | A, B, C ~ Normal(
        1.25 + 2.50 sin(A) + 0.75 B² - 1.50 C,
        0.20²
    )
    """
    return 1.25 + 2.50 * np.sin(X["A"]) + 0.75 * X["B"] ** 2 - 1.50 * X["C"]


@pytest.fixture(scope="module")
def nonlinear_data():
    rng = np.random.default_rng(seed=42)

    X_train = pd.DataFrame(
        {
            "A": rng.uniform(-2.5, 2.5, size=1000),
            "B": rng.uniform(-2.5, 2.5, size=1000),
            "C": rng.uniform(-2.5, 2.5, size=1000),
        },
        index=pd.RangeIndex(0, 1_000),
    )

    X_test = pd.DataFrame(
        {
            "A": rng.uniform(-2.5, 2.5, size=200),
            "B": rng.uniform(-2.5, 2.5, size=200),
            "C": rng.uniform(-2.5, 2.5, size=200),
        },
        index=pd.RangeIndex(2_000, 2_200),
    )

    y_train = pd.DataFrame(
        {
            "Y": (
                _conditional_mean(X_train)
                + rng.normal(
                    loc=0.0,
                    scale=0.20,
                    size=len(X_train),
                )
            )
        },
        index=X_train.index,
    )

    expected_test_mean = _conditional_mean(X_test)

    return X_train, y_train, X_test, expected_test_mean


@pytest.fixture(scope="module")
def fitted_parameter(nonlinear_data):
    if not _check_soft_dependencies("pygam", severity="none"):
        pytest.skip("execute only if required dependency present")

    X_train, y_train, _, _ = nonlinear_data

    parameter = SkproAdapter(
        estimator=GAMRegressor(
            max_iter=200,
            tol=1e-4,
        )
    )
    parameter.fit(X_train, y_train)

    return parameter


class TestSkproAdapter: ...


@pytest.mark.skipif(
    not _check_soft_dependencies("pygam", severity="none"),
    reason="execute only if required dependency present",
)
class TestSkproAdapterPygam:
    def test_base_parameter_metadata(self):
        estimator = GAMRegressor()
        parameter = SkproAdapter(estimator=estimator)

        assert parameter.__class__.__name__ == "SkproAdapter"
        assert parameter.estimator is estimator
        assert parameter.estimator_ is estimator
        assert parameter._get_delegate() is estimator
        assert parameter.get_params(deep=False)["estimator"] is estimator

        assert parameter.get_class_tag("variable_type") == "continuous"
        assert parameter.get_class_tag("produces_factor") is False
        assert parameter.get_class_tag("is_linear_gaussian") is False
        assert parameter.get_class_tag("missing") is False
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("can_be_root") is False
        assert parameter.get_class_tag("python_dependencies") == ()

    def test_fit(
        self,
        nonlinear_data,
    ):
        X_train, y_train, _, _ = nonlinear_data

        parameter = SkproAdapter(estimator=GAMRegressor(max_iter=200, tol=1e-4))
        parameter.fit(X_train, y_train)

        assert parameter.is_fitted is True
        assert hasattr(parameter, "estimator_")

        np.testing.assert_array_equal(
            parameter.feature_names_in_,
            X_train.columns,
        )

    def test_predict_proba(
        self,
        nonlinear_data,
        fitted_parameter,
    ):
        _, _, X_test, _ = nonlinear_data
        parameter = fitted_parameter

        results = parameter.predict_proba(X_test)

        assert isinstance(results, Normal)
        assert results.index.equals(X_test.index)

        pd.testing.assert_index_equal(
            results.mean().index,
            X_test.index,
        )
        pd.testing.assert_index_equal(
            results.mean().columns,
            pd.Index(["Y"]),
        )

    def test_predict_proba_recovers_nonlinear_mean_and_noise(
        self,
        nonlinear_data,
        fitted_parameter,
    ):
        _, _, X_test, expected_test_mean = nonlinear_data
        parameter = fitted_parameter

        pred = parameter.predict_proba(X_test)
        pred_mean = pred.mean()["Y"].to_numpy()
        pred_var = pred.var()["Y"].to_numpy()
        expected = expected_test_mean.to_numpy()

        rmse = np.sqrt(np.mean((pred_mean - expected) ** 2))
        assert rmse < 0.30

        assert np.isfinite(pred_mean).all()
        assert np.isfinite(pred_var).all()
        assert (pred_var > 0).all()

    def test_identifiable_feature_effects(
        self,
        fitted_parameter,
    ):
        parameter = fitted_parameter

        diagnostic_X = pd.DataFrame(
            {
                "A": [
                    0.0,
                    np.pi / 2,
                    -np.pi / 2,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "B": [
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "C": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.5,
                    -1.5,
                ],
            },
            index=pd.Index(
                [
                    "baseline",
                    "A_high",
                    "A_low",
                    "B_high",
                    "B_base",
                    "C_high",
                    "C_low",
                ]
            ),
        )

        pred_mean = parameter.predict_proba(diagnostic_X).mean()["Y"]
        expected = _conditional_mean(diagnostic_X)

        np.testing.assert_allclose(
            pred_mean.to_numpy(),
            expected.to_numpy(),
            atol=0.45,
        )

        assert pred_mean["A_high"] > pred_mean["baseline"] > pred_mean["A_low"]
        assert pred_mean["B_high"] > pred_mean["B_base"]
        assert pred_mean["C_low"] > pred_mean["baseline"] > pred_mean["C_high"]

    def test_predict_proba_rejects_reordered_features(
        self,
        nonlinear_data,
        fitted_parameter,
    ):
        _, _, X_test, _ = nonlinear_data
        parameter = fitted_parameter

        with pytest.raises(ValueError, match="same columns"):
            parameter.predict_proba(X_test.loc[:, ["C", "A", "B"]])
