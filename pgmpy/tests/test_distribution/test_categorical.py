import numpy as np

from pgmpy.distribution.categorical import CategoricalDistribution


class TestCategoricalDistribution:
    """Tests for Categorical distributions."""

    def test_default(self):
        dist = CategoricalDistribution()

        assert dist.name == "CategoricalDistribution"
        assert dist.get_class_tag("python_version") is None
        assert dist.get_class_tag("python_dependencies") is None
        assert dist.get_class_tag("distr:measuretype") == "discrete"
        assert dist.get_class_tag("distr:paramtype") == "parametric"
        assert dist.get_class_tag("capabilities:approx") == []
        assert dist.get_class_tag("capabilities:exact") == [
            "log_pmf",
            "pmf",
            "cdf",
            "ppf",
        ]
        assert dist.get_class_tag("broadcast_init") == "off"

    def test_interface_compatibility(self):
        """ensure interface compatibility by skpro.utils.estimator_checks.check_estimator"""
        from skpro.utils.estimator_checks import check_estimator

        values = [[0.1, 0.9], [0.7, 0.3]]
        values = np.asarray(values)

        values = [[0.1, 0.9], [0.7, 0.3]]
        values = np.asarray(values)
        x = [["A"], ["B"]]
        x = np.asarray(x)

        dist = CategoricalDistribution(values=values, state_names=["A", "B"])

        check_estimator(dist, raise_exceptions=True, verbose=False)

    def test_init(self):
        """test"""
        ...

    def test_mean(self):
        """test"""
        ...

    def test_var(self):
        """test"""
        ...

    def test_cdf(self):
        """test"""
        ...

    def test_ppf(self):
        """test"""
        ...

    def test_pmf(self):
        """test"""
        ...

    def test_log_pmf(self):
        """test"""
        ...

    def test_sample(self):
        """test"""
        ...
