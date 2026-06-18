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

    def test_root_node_case(self, discrete_data):
        _, y = discrete_data
        parameter = TabularCPD()
        parameter.fit(y)

        assert parameter.values_ == ...

    def test_child_node_case(self, discrete_data):
        X, y = discrete_data
        parameter = TabularCPD()
        parameter.fit(X, y)

        assert parameter.values_ == ...

    def test_fit(self, discrete_data):
        X, y = discrete_data
        parameter = TabularCPD()

        parameter.fit(X, y)

        assert parameter.is_fitted_ is True
        assert parameter.estimator_.__class__.__name__ == "DiscreteMLE"
        assert parameter._label_binarizer.__class__.__name__ == "LabelBinarizer"
        assert parameter.values_ == ...
        assert parameter.index_ == ...
        assert parameter.state_names_ == ...
        assert parameter.columns_ == ...

    def test_predict_proba(self, discrete_data):
        X, y = discrete_data
        parameter = TabularCPD()

        parameter.predict_proba(X)

        with pytest.raises(RuntimeError):
            parameter = TabularCPD()
            parameter.predict_proba(X)

    def test_set_values(self):
        parameter = TabularCPD()

        parameter.set_values(...)

        assert parameter.values_ == ...
        assert parameter.index_ == ...
        assert parameter.state_names_ == ...
        assert parameter.columns_ == ...

    def test_get_values(self):
        parameter = TabularCPD()

        parameter.set_values(...)
        result = parameter.get_values(...)

        assert result["values"] == ...
        assert result["state_names"] == ...
        assert result["index"] == ...
        assert result["columns"] == ...
