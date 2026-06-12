import numpy as np
import pandas as pd
import pytest

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
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = [1, 2, "C"]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        assert dist.values == values
        assert dist.state_names == state_names
        assert dist.columns == ["variable"]
        np.testing.assert_allclose(dist._values, np.asarray(values, dtype=float))
        np.testing.assert_array_equal(dist._state_names, np.asarray(state_names))

        # Case 1: values: list
        values = [[0.1, 0.9], [0.7, 0.3]]
        state_names = [1, 2]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        assert dist.values == values
        assert dist.state_names == state_names
        assert dist.columns == ["variable"]
        np.testing.assert_allclose(dist._values, np.asarray(values, dtype=float))
        np.testing.assert_array_equal(dist._state_names, np.asarray(state_names))

        # Case 2: values: numpy
        values = [[0.1, 0.9], [0.7, 0.3]]
        values = np.asarray(values, dtype=float)
        state_names = [1, 2]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        assert dist.state_names == state_names
        assert dist.columns == ["variable"]
        np.testing.assert_allclose(dist._values, np.asarray(values, dtype=float))
        np.testing.assert_array_equal(dist._state_names, np.asarray(state_names))

        # Case 3: state_names: str
        values = [[0.1, 0.9], [0.7, 0.3]]
        state_names = ["A", "B"]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        assert dist.values == values
        assert dist.state_names == state_names
        assert dist.columns == ["variable"]
        np.testing.assert_allclose(dist._values, np.asarray(values, dtype=float))
        np.testing.assert_array_equal(dist._state_names, np.asarray(state_names))

        # Case 4: state_names: int
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        assert dist.values == values
        assert dist.state_names == state_names
        assert dist.columns == ["variable"]
        np.testing.assert_allclose(dist._values, np.asarray(values, dtype=float))
        np.testing.assert_array_equal(dist._state_names, np.asarray(state_names))

        # Case 5: index
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = ["A", "B", "C"]

        dist = CategoricalDistribution(values=values, state_names=state_names, index=["studentA", "studentB"])

        assert dist.values == values
        assert dist.state_names == state_names
        assert list(dist.index) == ["studentA", "studentB"]
        assert list(dist.columns) == ["variable"]
        np.testing.assert_allclose(dist._values, np.asarray(values, dtype=float))
        np.testing.assert_array_equal(dist._state_names, np.asarray(state_names))

        # Case 6: columns
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = ["A", "B", "C"]

        dist = CategoricalDistribution(
            values=values, state_names=state_names, index=["studentA", "studentB"], columns=["grade"]
        )

        assert dist.values == values
        assert dist.state_names == state_names
        assert list(dist.index) == ["studentA", "studentB"]
        assert list(dist.columns) == ["grade"]
        np.testing.assert_allclose(dist._values, np.asarray(values, dtype=float))
        np.testing.assert_array_equal(dist._state_names, np.asarray(state_names))

        # Case 7: wrong values
        values = [[0.1, 0.2, 0.9], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]

        with pytest.raises(ValueError):
            dist = CategoricalDistribution(values=values, state_names=state_names)

        # Case 8: wrong state_names
        values = [[0.1, 0.2, 0.8], [0.5, 0.3, 0.2]]
        state_names = [1, 1, 1]

        with pytest.raises(ValueError):
            dist = CategoricalDistribution(values=values, state_names=state_names)

        # Case 9: wrong index
        values = [[0.1, 0.2, 0.8], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]
        index = ["A", "B", "C"]

        with pytest.raises(ValueError):
            dist = CategoricalDistribution(values=values, state_names=state_names, index=index)

        # Case 10: wrong columns
        values = [[0.1, 0.2, 0.8], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]
        columns = ["A", "B", "C"]

        with pytest.raises(ValueError):
            dist = CategoricalDistribution(values=values, state_names=state_names, columns=columns)

        # Case 13: worng shape(values, state_names)
        values = [[0.1, 0.2], [0.5, 0.3]]
        state_names = [1, 2, 3]

        with pytest.raises(ValueError):
            dist = CategoricalDistribution(values=values, state_names=state_names)

    def test_cdf(self):
        """test"""
        # Case 1: x: int
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]
        x = [[1], [1]]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        expected = pd.DataFrame({"variable": [0.1, 0.5]})
        pd.testing.assert_frame_equal(dist.cdf(x), expected)

        # Case 2: x: str
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = ["A", "B", "C"]
        x = [["A"], ["C"]]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        expected = pd.DataFrame({"variable": [0.1, 1]})
        pd.testing.assert_frame_equal(dist.cdf(x), expected)

        # Case 3: wrong x's ndim
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]
        x = [1, 1]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        with pytest.raises(ValueError):
            dist.cdf(x)

        # Case 4: wrong x's value
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]
        x = [["A"], ["B"]]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        with pytest.raises(ValueError):
            dist.cdf(x)

        # Case 5: broadcasting
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]
        x = [[1]]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        expected = pd.DataFrame({"variable": [0.1, 0.5]})
        pd.testing.assert_frame_equal(dist.cdf(x), expected)

    def test_ppf(self):
        """test"""
        # Case 1: p: float
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]
        p = [[0.1], [0.9]]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        expected = pd.DataFrame({"variable": [1, 3]})
        pd.testing.assert_frame_equal(dist.ppf(p), expected)

        # Case 2: p: np.float
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]
        p = [[0.1], [0.9]]
        p = np.asarray(p, dtype=float)

        dist = CategoricalDistribution(values=values, state_names=state_names)

        expected = pd.DataFrame({"variable": [1, 3]})
        pd.testing.assert_frame_equal(dist.ppf(p), expected)

        # Case 3: wrong p's ndim
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]
        p = [0.1, 0.9]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        with pytest.raises(ValueError):
            dist.ppf(p)

        # Case 4: wrong p's value
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]
        p = [[-0.1], [1.1]]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        with pytest.raises(ValueError):
            dist.ppf(p)

        # Case 5: broadcasting
        values = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]
        state_names = [1, 2, 3]
        p = [[0.1]]

        dist = CategoricalDistribution(values=values, state_names=state_names)

        expected = pd.DataFrame({"variable": [1, 1]})
        pd.testing.assert_frame_equal(dist.ppf(p), expected)

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
