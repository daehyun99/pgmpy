import pandas as pd
import pytest
from sklearn.datasets import make_moons

from pgmpy.parameter._base import BaseParameter


class TestBaseParameter:
    """Test BaseParameter class"""

    def test_base_parameter_default(self):
        parameter = BaseParameter()

        assert parameter.__class__.__name__ == "BaseParameter"
        assert parameter.get_class_tag("variable_type") is str
        assert parameter.get_class_tag("produces_factor") is bool
        assert parameter.get_class_tag("missing") is bool
        assert parameter.get_class_tag("is_linear_gaussian") is bool
        assert parameter.get_class_tag("supports_fit_joint") is bool
        assert parameter.get_class_tag("python_dependencies") is tuple

    def test_fit(self):
        X_arr, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
        X = pd.DataFrame(X_arr[:, 0].reshape(-1, 1), columns=["x"])
        y = pd.DataFrame(X_arr[:, 1].reshape(-1, 1), columns=["y"])
        parameter = BaseParameter()

        with pytest.raises(NotImplementedError):
            parameter.fit(X, y)

    def test_predict_proba(self):
        X_arr, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
        X = pd.DataFrame(X_arr[:, 0].reshape(-1, 1), columns=["x"])
        parameter = BaseParameter()

        with pytest.raises(NotImplementedError):
            parameter.predict_proba(X)
