import numpy as np
import pandas as pd
import pytest
from skpro.distributions.normal import Normal
from skpro.distributions.poisson import Poisson

from pgmpy.parameter.adapter.DistributionAdapter import DistributionAdapter


class TestDistributionAdapter:
    def test_base_parameter_default(self):
        distribution = Normal(mu=1.5, sigma=2.0)
        parameter = DistributionAdapter(distribution=distribution)

        assert parameter.__class__.__name__ == "DistributionAdapter"
        assert parameter.distribution is distribution
        assert parameter.get_params(deep=False)["distribution"] is distribution

        assert parameter.get_class_tag("parameter_type") == "distribution"
        assert parameter.get_class_tag("produces_factor") is False
        assert parameter.get_class_tag("is_linear_gaussian") is False
        assert parameter.get_class_tag("missing") is False
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("can_be_root") is True

    @pytest.mark.parametrize(
        ("distribution", "expected_type", "expected_mean"),
        [
            (Normal(mu=1.5, sigma=2.0), Normal, 1.5),
            (Poisson(mu=3.0), Poisson, 3.0),
        ],
    )
    def test_predict_proba_broadcasts_distribution(self, distribution, expected_type, expected_mean):
        X = pd.DataFrame({"A": [0.0, 1.0, 2.0]}, index=pd.Index(["a", "b", "c"]))
        y = pd.DataFrame({"Y": [1.0, 2.0, 3.0]}, index=X.index)
        parameter = DistributionAdapter(distribution=distribution)

        parameter.fit(X, y)
        result = parameter.predict_proba(X)

        assert parameter.is_fitted is True
        assert isinstance(result, expected_type)
        assert result.index.equals(X.index)
        assert result.columns.equals(pd.Index(["Y"]))
        np.testing.assert_allclose(result.mean().to_numpy(), expected_mean)

    def test_rejects_non_skpro_distribution(self):
        with pytest.raises(TypeError, match="skpro distribution"):
            DistributionAdapter(distribution="not a distribution")
