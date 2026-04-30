from __future__ import annotations

from collections.abc import Callable

from .base import BaseParameterEstimator


class HybridEstimator(BaseParameterEstimator):
    """Orchestrates parameter fitting for mixed CPD/model objects in a Bayesian network.

    The estimator expects a model that stores CPD-like objects in ``model.cpds`` and each
    CPD-like object to expose a ``fit`` method. This allows combining discrete
    ``TabularCPD`` objects with continuous/skpro-style regressors in a single model.

    Parameters
    ----------
    fit_dispatch: dict, optional
        Mapping of CPD/model types to custom fit callables. Callable signature must be
        ``callable(cpd, data)`` and should return the fitted CPD/model object. If a CPD's
        type is not present in this mapping, ``cpd.fit(data)`` is used.
    """

    _tags = {
        "supported_model_types": (),
        "supports_weighted_data": False,
    }

    def __init__(self, fit_dispatch: dict[type, Callable] | None = None) -> None:
        self.fit_dispatch = fit_dispatch
        super().__init__()

    def _validate_inputs(self, model, data, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("HybridEstimator does not support `sample_weight`.")

        if not hasattr(model, "cpds"):
            raise ValueError("model must define a `cpds` attribute containing CPD-like objects.")

        if len(model.cpds) == 0:
            raise ValueError("model.cpds is empty. Add CPD-like objects before fitting.")

        missing_fit = [cpd for cpd in model.cpds if not hasattr(cpd, "fit")]
        if missing_fit:
            raise ValueError("All objects in model.cpds must implement a `fit` method.")

        return model, None

    def _fit_single(self, cpd, data):
        dispatch = self.fit_dispatch if isinstance(self.fit_dispatch, dict) else {}
        fit_fn = next((fn for cpd_type, fn in dispatch.items() if isinstance(cpd, cpd_type)), None)

        if fit_fn is not None:
            return fit_fn(cpd, data)

        cpd.fit(data)
        return cpd

    def fit(self, model, data, sample_weight=None):
        """Fit all CPD-like objects in ``model.cpds`` on ``data``.

        Parameters
        ----------
        model: object
            A Bayesian-network-like model object with a ``cpds`` list.

        data: pandas.DataFrame
            Training data used by each CPD/model object's ``fit`` implementation.

        sample_weight: optional
            Not supported.

        Returns
        -------
        self
            Fitted estimator with learned parameters in ``parameters_``.
        """
        self._model, _ = self._validate_inputs(model, data, sample_weight=sample_weight)
        self._data = data

        self.parameters_ = [self._fit_single(cpd, data) for cpd in self._model.cpds]
        return self
