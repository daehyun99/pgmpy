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
            "mean",
            "var",
            "log_pmf",
            "pmf",
            "cdf",
            "ppf",
        ]
        assert dist.get_class_tag("broadcast_init") == "on"

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
