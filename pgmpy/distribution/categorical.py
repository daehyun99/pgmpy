"""
- do not write to reserved attributes: index, columns, head, tail, loc, iloc, at, iat,
  shape, ndim, _bc_params, _tags, _tags_dynamic, _config, _config_dynamic
"""

import numpy as np
import pandas as pd
from skpro.distributions.base import BaseDistribution


# todo: change class name and write docstring
class CategoricalDistribution(BaseDistribution):
    """Categorical probability distribution.

    todo: describe your custom probability distribution here

    Parameters
    ----------
    parama : float or np.ndarray
        descriptive explanation of parama
    paramb : float or np.ndarray, optional (default='default')
        descriptive explanation of paramb
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

    def __init__(self, values, state_names=None, index=None, columns=["variable"]):

        self.values = values
        self.state_names = state_names

        values = np.asarray(values, dtype=float)

        n_conditions, n_states = values.shape

        if state_names is None:
            state_names = np.arange(n_states)
        else:
            state_names = np.asarray(state_names)

        if index is None:
            index = pd.RangeIndex(n_conditions)

        self.values = values
        self.state_names = state_names

        super().__init__(index=index, columns=columns)

    def _pmf(self, x):
        """Probability mass function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pmf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pmf values at the given points
        """
        values = self.values
        state_names = self.state_names

        x = np.asarray(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if x.shape != self.shape:
            raise ValueError(f"x must have shape {self.shape}, but got {x.shape}")

        x_flat = x[:, 0]

        matches = x_flat[:, None] == state_names[None, :]

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
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pmf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            log pmf values at the given points
        """
        pmf = self._pmf(x)
        return np.log(pmf)

    def _cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the cdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            cdf values at the given points
        """
        values = self.values
        state_names = self.state_names

        x = np.asarray(x)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if x.shape != self.shape:
            raise ValueError(f"x must have shape {self.shape}, but got {x.shape}")

        x_flat = x[:, 0]

        res = np.empty(values.shape[0], dtype=float)

        for i, xi in enumerate(x_flat):
            mask = state_names <= xi
            res[i] = values[i, mask].sum()

        return res.reshape(-1, 1)

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf.

        Parameters
        ----------
        p : 2D np.ndarray, same shape as ``self``
            values to evaluate the ppf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            ppf values at the given points
        """
        values = self.values
        state_names = self.state_names

        p = np.asarray(p, dtype=float)

        if p.ndim == 0:
            p = p.reshape(1, 1)

        if p.ndim == 1:
            p = p.reshape(-1, 1)

        if p.shape != self.shape:
            raise ValueError(f"p must have shape {self.shape}, but got {p.shape}")

        if np.any((p < 0) | (p > 1)):
            raise ValueError("p values must be between 0 and 1.")

        p_flat = p[:, 0]

        # Ensure quantiles follow the same ordering implied by CDF:
        # F(x) = P(X <= x)
        order = np.argsort(state_names)
        sorted_states = state_names[order]
        sorted_values = values[:, order]

        cdf = np.cumsum(sorted_values, axis=1)

        # Numerical safety: if probabilities are normalized but floating error exists
        cdf[:, -1] = 1.0

        idx = np.array([np.searchsorted(cdf[i], p_flat[i], side="left") for i in range(values.shape[0])])

        res = sorted_states[idx]

        return res.reshape(-1, 1)

    # todo: consider implementing
    # at least one of _ppf and _sample must be implemented
    # if not implemented, uses _ppf for sampling (inverse cdf on uniform)
    # def _sample(self, n_samples=None):
    #     """Sample from the distribution.

    #     Parameters
    #     ----------
    #     n_samples : int, optional, default = None
    #         number of samples to draw from the distribution

    #     Returns
    #     -------
    #     pd.DataFrame
    #         samples from the distribution

    #         * if ``n_samples`` is ``None``:
    #         returns a sample that contains a single sample from ``self``,
    #         in ``pd.DataFrame`` mtype format convention, with ``index`` and ``columns``
    #         as ``self``
    #         * if n_samples is ``int``:
    #         returns a ``pd.DataFrame`` that contains ``n_samples`` i.i.d.
    #         samples from ``self``, in ``pd-multiindex`` mtype format convention,
    #         with same ``columns`` as ``self``, and row ``MultiIndex`` that is product
    #         of ``RangeIndex(n_samples)`` and ``self.index``
    #     """
    #     param1 = self._bc_params["param1"]  # returns broadcast params to x.shape
    #     param2 = self._bc_params["param2"]  # returns broadcast params to x.shape

    #     res = "do_sth_with(" + param1 + param2 + ")"  # replace this by internal logic
    #     return res
    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator

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

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        #
        # this can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from skpro or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params
        # params = {"est": value3, "parama": value4}
        # return params
