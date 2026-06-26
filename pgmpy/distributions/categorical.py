import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from skpro.distributions.base import BaseDistribution


class CategoricalDistribution(BaseDistribution):
    """Categorical distribution for discrete random variables.

    Represents one or more categorical probability distributions over a
    finite set of discrete states. Each row of ``probs`` defines the
    probability mass assigned to the states specified by ``categories``.

    Parameters
    ----------
    probs : array-like of shape (n_instances, n_states)
        Probability masses for each state. Each row represents one
        categorical distribution and must contain non-negative probs
        that sum to 1.
    categories : array-like of shape (n_states,)
        Names or labels of the possible discrete states. The order of
        ``categories`` corresponds to the order of probabilities in
        each row of ``probs``. State names must be unique.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = ["variable"]

    Examples
    --------
    >>> from pgmpy.distributions.categorical import CategoricalDistribution

    >>> probs = [[0.2, 0.4, 0.3, 0.1], [0.4, 0.4, 0.1, 0.1]]
    >>> categories = ["A", "B", "C", "D"]
    >>> index=["studentA", "studentB"]
    >>> columns = ["grade"]
    >>> dist = CategoricalDistribution(probs=probs, categories=categories, index=index, columns=columns)

    """

    _tags = {
        "python_version": None,
        "python_dependencies": None,
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "capabilities:approx": [],
        "capabilities:exact": ["log_pmf", "pmf", "cdf", "ppf"],
        "broadcast_init": "off",
    }

    def __init__(self, probs, categories, index=None, columns=None):

        self.probs = probs
        self.categories = categories

        # Validate probs.
        probs_for_check = np.asarray(probs, dtype=float)
        if np.any(probs_for_check < 0):
            raise ValueError("probs must contain only non-negative probabilities")

        row_sums = probs_for_check.sum(axis=1)
        invalid_rows = np.flatnonzero(~np.isclose(row_sums, 1.0, rtol=1e-9, atol=1e-12))
        if invalid_rows.size:
            raise ValueError(
                "The probabilities in each row of probs must sum to 1; "
                f"invalid row indices: {invalid_rows.tolist()}, "
                f"row sums: {row_sums[invalid_rows].tolist()}"
            )

        # Validate categories.
        expected_type = type(categories[0])
        mismatched_items = [
            (index, value, type(value).__name__)
            for index, value in enumerate(categories)
            if type(value) is not expected_type
        ]

        if mismatched_items:
            raise TypeError(
                "All probs in categories must have the same type; "
                f"expected {expected_type.__name__}, "
                f"but found mismatched probs: {mismatched_items}"
            )

        # Validate shape of state_name and probs.
        if len(categories) != len(set(categories)):
            raise ValueError(f"""Categories must contain unique probs: {categories}""")

        if len(probs[0]) != len(categories):
            raise ValueError(
                f"""mismatch between the shape of categories and probs : {len(probs[0])}, {len(categories)}"""
            )

        # Validate index, columns.
        if index is None:
            index = pd.RangeIndex(len(probs))
        elif len(index) != len(probs):
            raise ValueError(f"The length of index must match the number of rows in probs : {len(index)}, {len(probs)}")
        if columns is None:
            columns = ["variable"]
        elif len(columns) != 1:
            raise ValueError("columns must contain exactly one column name")

        super().__init__(index=index, columns=columns)

    def _pmf(self, x):
        """Probability mass function.

        Parameters
        ----------
        x : 2D np.ndarray

        Returns
        -------
        2D np.ndarray

        """
        probs = np.asarray(self.probs, dtype=float)
        categories = np.asarray(self.categories)

        matches = x == categories

        valid = matches.any(axis=1)

        state_idx = matches.argmax(axis=1)
        row_idx = np.arange(probs.shape[0])

        res = probs[row_idx, state_idx]

        res = np.where(valid, res, 0.0)

        return res.reshape(-1, 1)

    def _log_pmf(self, x):
        """Logarithmic probability mass function.

        Parameters
        ----------
        x : 2D np.ndarray

        Returns
        -------
        2D np.ndarray

        """
        pmf = self._pmf(x)
        return np.log(pmf)

    def _cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : 2D np.ndarray

        Returns
        -------
        2D np.ndarray

        """
        probs = np.asarray(self.probs, dtype=float)
        categories = np.asarray(self.categories)

        res = np.empty(probs.shape[0], dtype=float)

        for i, xi in enumerate(x):
            target = xi[0]

            idx_arr = np.where(categories == target)[0]

            if len(idx_arr) == 0:
                res[i] = 0.0
            else:
                idx = idx_arr[0]
                res[i] = probs[i, : idx + 1].sum()

        return res.reshape(-1, 1)

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf.

        Parameters
        ----------
        p : 2D np.ndarray

        Returns
        -------
        2D np.ndarray

        """
        probs = np.asarray(self.probs, dtype=float)
        categories = np.asarray(self.categories)

        if np.any((p < 0) | (p > 1)):
            raise ValueError("p values must be between 0 and 1.")

        res = np.empty(probs.shape[0], dtype=categories.dtype)

        for i, pi in enumerate(p):
            prob = pi[0]

            cum_probs = np.cumsum(probs[i])
            idx = np.searchsorted(cum_probs, prob, side="left")

            if idx >= probs.shape[1]:
                idx = probs.shape[1] - 1

            res[i] = categories[idx]

        return res.reshape(-1, 1)

    def _sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None

        Returns
        -------
        pd.DataFrame

        """
        probs = self.probs
        categories = self.categories

        probs = np.asarray(probs, dtype=float)
        categories = np.asarray(categories)

        if n_samples is None:
            n_samples = 1
            single_sample = True
        else:
            single_sample = False

        n_rows, _ = probs.shape

        sampled = np.empty((n_samples, n_rows), dtype=categories.dtype)

        for i in range(n_rows):
            sampled[:, i] = np.random.choice(
                categories,
                size=n_samples,
                p=probs[i],
                replace=True,
            )

        index = self.index
        columns = self.columns

        if single_sample:
            res = pd.DataFrame(
                sampled[0].reshape(-1, 1),
                index=index,
                columns=columns,
            )
        else:
            multi_index = pd.MultiIndex.from_product(
                [pd.RangeIndex(n_samples), index],
                names=["sample", None],
            )

            res = pd.DataFrame(
                sampled.reshape(n_samples * n_rows, 1),
                index=multi_index,
                columns=columns,
            )

        return res

    def plot(self, fun="pmf", ax=None, **kwargs):
        """Plot the categorical probability mass function.

        A separate bar plot is created for each row in ``probs``. The category
        labels are taken from ``categories``, and the height of each bar represents
        the corresponding probability.

        Each subplot is labeled using the corresponding entry in ``index``. The
        first entry in ``columns`` is used as the figure title.

        Parameters
        ----------
        fun : {"pmf"}, default="pmf"
            Distribution function to plot.
            Currently, only the probability mass function (``"pmf"``) is supported.
        ax : matplotlib Axes object, optional
            matplotlib Axes to plot in
            if not provided, defaults to current axes (``plot.gca``)
        kwargs : keyword arguments
            passed to the plotting function

        Returns
        -------
        fig : matplotlib.Figure, only returned if self is array distribution
            matplotlig Figure object for subplots
        ax : matplotlib.Axes
            the axis or axes on which the plot is drawn

        Notes
        -----
        The `matplotlib` library must be installed to use this method.

        Examples
        --------
        >>> probs = [[0.2, 0.4, 0.3, 0.1], [0.4, 0.4, 0.1, 0.1]]
        >>> categories = ["A", "B", "C", "D"]
        >>> index = ["studentA", "studentB"]
        >>> columns = ["grade"]
        >>> dist = CategoricalDistribution(probs=probs, categories=categories, index=index, columns=columns)
        >>> dist.plot(fun="pmf") # doctest: +SKIP
        <Figure ...>
        """
        _check_soft_dependencies("matplotlib", obj="distribution plot")
        import matplotlib.pyplot as plt

        probs = np.asarray(self.probs, dtype=float)
        states = np.asarray(self.categories)

        n_rows, _ = probs.shape

        if fun != "pmf":
            raise NotImplementedError("`Categorical` only supports `pmf` currently")

        if ax is None:
            fig, axes = plt.subplots(
                n_rows,
                1,
                squeeze=False,
                sharex=True,
                sharey=True,
            )
        else:
            axes = np.asarray(ax, dtype=object).reshape(n_rows, 1)
            fig = axes[0, 0].figure

        for i in range(n_rows):
            current_ax = axes[i, 0]
            current_ax.bar(states, probs[i], **kwargs)
            current_ax.set_ylabel(str(self.index[i]))
            current_ax.set_ylim(0, 1)

        axes[0, 0].set_title(str(self.columns[0]))
        axes[-1, 0].set_xlabel("state names")

        return fig, axes[:, 0]

    def _subset_params(self, rowidx, colidx, coerce_scalar=False):
        """Subset distribution parameters to given rows and columns.

        Returns
        -------
        dict

        """
        probs = np.asarray(self.probs, dtype=float)
        categories = np.asarray(self.categories)

        if rowidx is not None:
            probs = probs[rowidx, :]

            if probs.ndim == 1:
                probs = probs.reshape(1, -1)

            if categories.ndim == 2:
                categories = categories[rowidx, :]
                if categories.ndim == 1:
                    categories = categories.reshape(1, -1)

        return {
            "probs": probs,
            "categories": categories,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {"probs": [[0.1, 0.9], [0.7, 0.3]], "categories": [1, 2]}
        params2 = {"probs": [[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]], "categories": [1, 2, 3]}
        return [params1, params2]
