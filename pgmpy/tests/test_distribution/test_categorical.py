from pgmpy.distribution.categorical import CategoricalDistribution


class TestCategoricalDistribution:
    """Tests for Categorical distributions."""

    def test_default(self):
        values = [[0.1, 0.9], [0.7, 0.3]]
        dist = CategoricalDistribution(values)

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

        dist = CategoricalDistribution(values=values, state_names=[1, 2])

        check_estimator(dist, raise_exceptions=True, verbose=False)

    def test_init(self):
        """test"""
        # Case 0: default

        # Case 1: values: list

        # Case 2: values: numpy

        # Case 3: state_names: str

        # Case 4: state_names: int

        # Case 5: index

        # Case 6: columns

        # Case 7: wrong values

        # Case 8: wrong state_names

        # Case 9: wrong index

        # Case 10: wrong columns

        # Case 11: values: pd.Series

        # Case 12: values: pd.DataFrame

        # Case 13: worng shape(values, state_names)

        # Case 14: state_names's order

        # Case 15: duplicate of state_names

    def test_cdf(self):
        """test"""
        # Case 1: x: int

        # Case 2: x: str

        # Case 3: wrong x's ndim

        # Case 4: wrong x's type

        # Case 5: wrong x's value

        # Case 6: wrong x's shape

        # Case 7: broadcasting

    def test_ppf(self):
        """test"""
        # Case 1: p: float

        # Case 2: p: np.float

        # Case 3: wrong p's ndim

        # Case 4: wrong p's type

        # Case 5: wrong p's value

        # Case 6: wrong p's shape

        # Case 7: broadcasting

    def test_pmf(self):
        """test"""
        # Case 1: x: int

        # Case 2: x: str

        # Case 3: wrong x's ndim

        # Case 4: wrong x's type

        # Case 5: wrong x's value

        # Case 6: wrong x's shape

        # Case 7: broadcasting

    def test_log_pmf(self):
        """test"""
        # Case 1: x: int

        # Case 2: x: str

        # Case 3: wrong x's ndim

        # Case 4: wrong x's type

        # Case 5: wrong x's value

        # Case 6: wrong x's shape

        # Case 7: broadcasting

    def test_sample(self):
        """test"""
        # Case 1: n_samples

        # Case 2: n_samples

        # Case 3: n_samples shape

        # Case 4: wrong n_samples value

    def test_mathematical_consistency(self):
        """test"""
        # Case 1: ppf(cdf(x)) = x 

        # Case 2: cdf(ppf(p)) = p

        # Case 3: log(pmf(x)) = log_pmf(x)
