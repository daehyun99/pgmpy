import numpy as np
import pandas as pd
import pytest

from pgmpy.parameter.TabularCPD import TabularCPD


@pytest.fixture
def discrete_data():
    rng = np.random.default_rng(seed=42)
    n_samples = 100

    X = pd.DataFrame(
        {
            "x1": rng.integers(0, 3, size=n_samples),  # {0, 1, 2}
            "x2": rng.integers(0, 2, size=n_samples),  # {0, 1}
        }
    )

    y = pd.DataFrame({"y": ((X["x1"] + X["x2"]) % 2).astype(int)})

    return X, y


class TestTabularCPD:
    """Test TabularCPD class"""

    def test_base_parameter_default(self):
        parameter = TabularCPD()

        assert parameter.__class__.__name__ == "TabularCPD"
        assert parameter.get_class_tag("variable_type") == "discrete"
        assert parameter.get_class_tag("produces_factor") is True
        assert parameter.get_class_tag("missing") is False
        assert parameter.get_class_tag("is_linear_gaussian") is False
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("python_dependencies") == ("skpro")

    def test_fit(self, discrete_data):
        # Case 1: root node case
        _, y = discrete_data
        parameter = TabularCPD()
        parameter.fit(y)

        assert parameter.is_fitted_ is True
        assert parameter.estimator_.__class__.__name__ == "TempMLE"  # "DiscreteMLE"
        assert parameter._label_binarizer.__class__.__name__ == "LabelBinarizer"
        np.testing.assert_allclose(
            parameter.CPT_,
            np.array(
                [
                    [0.42],
                    [0.58],
                ]
            ),
        )
        assert list(parameter.categories_) == [0, 1]
        assert parameter.columns_ == ["variable"]

        # Case 2: not root node case
        X, y = discrete_data
        parameter = TabularCPD()

        parameter.fit(X, y)

        expected_CPT = np.array(
            [
                [0.75, 0.50, 0.66666667, 0.00],
                [0.25, 0.50, 0.33333333, 1.00],
            ]
        )

        assert parameter.is_fitted_ is True
        assert parameter.estimator_.__class__.__name__ == "TempMLE"  # "DiscreteMLE"
        assert parameter._label_binarizer.__class__.__name__ == "LabelBinarizer"
        np.testing.assert_allclose(
            parameter.CPT_,
            expected_CPT,
            rtol=1e-7,
            atol=1e-8,
        )
        assert list(parameter.categories_) == [0, 1]
        assert parameter.columns_ == ["variable"]

    def test_predict_proba(self, discrete_data):
        X, y = discrete_data
        parameter = TabularCPD()
        parameter.fit(X, y)
        dist = parameter.predict_proba(X)

        expected = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

        np.testing.assert_array_equal(dist.probs[:5], expected)
        assert dist.__class__.__name__ == "CategoricalDistribution"
        assert list(dist.categories) == [0, 1]
        assert list(dist.columns) == ["variable"]

        with pytest.raises(RuntimeError):
            parameter = TabularCPD()
            parameter.predict_proba(X)

    def test_set_values(self):
        parameter = TabularCPD()

        assert hasattr(parameter, "CPT_") is False
        assert hasattr(parameter, "categories_") is False
        assert hasattr(parameter, "columns_") is False
        assert parameter.is_fitted_ is False

        parameter.set_values(
            values=np.array(
                [
                    [0.2, 0.3],
                    [0.8, 0.7],
                ]
            ),
            columns=["grade"],
            state_names=["P", "F"],
            evidence_states=pd.MultiIndex.from_tuples(
                [
                    (0, 0),
                    (0, 1),
                    (1, 0),
                    (1, 1),
                ],
                names=["x1", "x2"],
            ),
        )

        assert hasattr(parameter, "CPT_")
        assert hasattr(parameter, "categories_")
        assert hasattr(parameter, "columns_")
        assert hasattr(parameter, "evidence_states_")
        assert parameter.is_fitted_ is True

    def test_get_values(self):
        parameter = TabularCPD()

        CPT = np.array(
            [
                [0.2, 0.3],
                [0.8, 0.7],
            ]
        )
        columns = ["grade"]
        categories = ["P", "F"]
        evidence_states = pd.MultiIndex.from_tuples(
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ],
            names=["x1", "x2"],
        )

        parameter.set_values(
            CPT=CPT,
            columns=columns,
            categories=categories,
            evidence_states=evidence_states,
        )

        result = parameter.get_values()

        np.testing.assert_array_equal(result["CPT"], CPT)
        assert result["columns"] == columns
        assert result["categories"] == categories
        pd.testing.assert_index_equal(
            result["evidence_states"],
            evidence_states,
        )
