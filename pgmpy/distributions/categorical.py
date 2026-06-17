import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from skpro.distributions.base import BaseDistribution


class CategoricalDistribution(BaseDistribution):
    """Categorical distribution for discrete random variables.

    Represents one or more categorical probability distributions over a
    finite set of discrete states. Each row of ``values`` defines the
    probability mass assigned to the states specified by ``state_names``.

    Parameters
    ----------
    values : array-like of shape (n_instances, n_states)
        Probability masses for each state. Each row represents one
        categorical distribution and must contain non-negative values
        that sum to 1.
    state_names : array-like of shape (n_states,)
        Names or labels of the possible discrete states. The order of
        ``state_names`` corresponds to the order of probabilities in
        each row of ``values``. State names must be unique.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = ["variable"]

    Examples
    --------
    >>> from pgmpy.distributions.categorical import CategoricalDistribution

    >>> values = [[0.2, 0.4, 0.3, 0.1], [0.4, 0.4, 0.1, 0.1]]
    >>> state_names = ["A", "B", "C", "D"]
    >>> index=["studentA", "studentB"]
    >>> columns = ["grade"]
    >>> dist = CategoricalDistribution(values=values, state_names=state_names, index=index, columns=columns)

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

    def __init__(self, values, state_names, index=None, columns=None):

        self.values = values
        self.state_names = state_names

        # Validate values.
        values_for_check = np.asarray(values, dtype=float)
        if np.any(values_for_check < 0):
            raise ValueError("values must contain only non-negative probabilities")

        row_sums = values_for_check.sum(axis=1)
        invalid_rows = np.flatnonzero(~np.isclose(row_sums, 1.0, rtol=1e-9, atol=1e-12))
        if invalid_rows.size:
            raise ValueError(
                "The probabilities in each row of values must sum to 1; "
                f"invalid row indices: {invalid_rows.tolist()}, "
                f"row sums: {row_sums[invalid_rows].tolist()}"
            )

        # Validate state_names.
        expected_type = type(state_names[0])
        mismatched_items = [
            (index, value, type(value).__name__)
            for index, value in enumerate(state_names)
            if type(value) is not expected_type
        ]

        if mismatched_items:
            raise TypeError(
                "All values in state_names must have the same type; "
                f"expected {expected_type.__name__}, "
                f"but found mismatched values: {mismatched_items}"
            )

        # Validate shape of state_name and values.
        if len(state_names) != len(set(state_names)):
            raise ValueError(f"state_names must contain unique values: {state_names}")

        if len(values[0]) != len(state_names):
            raise ValueError(
                f"mismatch between the shape of state_names and values: {len(values[0])}, {len(state_names)}"
            )

        # Validate index, columns.
        if index is None:
            index = pd.RangeIndex(len(values))
        elif len(index) != len(values):
            raise ValueError(
                f"The length of index must match the number of rows in values: {len(index)}, {len(values)}"
            )
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
        values = self.values
        state_names = self.state_names

        values = np.asarray(values, dtype=float)
        state_names = np.asarray(state_names)

        matches = x == state_names

        valid = matches.any(axis=1)

        state_idx = matches.argmax(axis=1)
        row_idx = np.arange(values.shape[0])

        res = values[row_idx, state_idx]

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
        values = self.values
        state_names = self.state_names

        values = np.asarray(values, dtype=float)
        state_names = np.asarray(state_names)

        res = np.empty(values.shape[0], dtype=float)

        for i, xi in enumerate(x):
            target = xi[0]

            idx_arr = np.where(state_names == target)[0]

            if len(idx_arr) == 0:
                res[i] = 0.0
            else:
                idx = idx_arr[0]
                res[i] = values[i, : idx + 1].sum()

        return res.reshape(-1, 1)

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf.

        Parameters
        ----------
        x : 2D np.ndarray

        Returns
        -------
        2D np.ndarray

        """
        values = self.values
        state_names = self.state_names

        values = np.asarray(values, dtype=float)
        state_names = np.asarray(state_names)

        if np.any((p < 0) | (p > 1)):
            raise ValueError("p values must be between 0 and 1.")

        res = np.empty(values.shape[0], dtype=state_names.dtype)

        for i, pi in enumerate(p):
            prob = pi[0]

            cum_values = np.cumsum(values[i])
            idx = np.searchsorted(cum_values, prob, side="left")

            if idx >= values.shape[1]:
                idx = values.shape[1] - 1

            res[i] = state_names[idx]

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
        values = self.values
        state_names = self.state_names

        values = np.asarray(values, dtype=float)
        state_names = np.asarray(state_names)

        if n_samples is None:
            n_samples = 1
            single_sample = True
        else:
            single_sample = False

        n_rows, _ = values.shape

        sampled = np.empty((n_samples, n_rows), dtype=state_names.dtype)

        for i in range(n_rows):
            sampled[:, i] = np.random.choice(
                state_names,
                size=n_samples,
                p=values[i],
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

        A separate bar plot is created for each row in ``values``. The category
        labels are taken from ``state_names``, and the height of each bar represents
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
        >>> values = [[0.2, 0.4, 0.3, 0.1], [0.4, 0.4, 0.1, 0.1]]
        >>> state_names = ["A", "B", "C", "D"]
        >>> index = ["studentA", "studentB"]
        >>> columns = ["grade"]
        >>> dist = CategoricalDistribution(values=values, state_names=state_names, index=index, columns=columns)
        fig, axes = dist.plot(fun="pmf")

        """
        _check_soft_dependencies("matplotlib", obj="distribution plot")
        import matplotlib.pyplot as plt

        values = np.asarray(self.values, dtype=float)
        states = np.asarray(self.state_names)

        n_rows, _ = values.shape

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
            current_ax.bar(states, values[i], **kwargs)
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
        values = np.asarray(self.values, dtype=float)
        state_names = np.asarray(self.state_names)

        if rowidx is not None:
            values = values[rowidx, :]

            if values.ndim == 1:
                values = values.reshape(1, -1)

            if state_names.ndim == 2:
                state_names = state_names[rowidx, :]
                if state_names.ndim == 1:
                    state_names = state_names.reshape(1, -1)

        return {
            "values": values,
            "state_names": state_names,
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
        params1 = {"values": [[0.1, 0.9], [0.7, 0.3]], "state_names": [1, 2]}
        params2 = {"values": [[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]], "state_names": [1, 2, 3]}
        return [params1, params2]
