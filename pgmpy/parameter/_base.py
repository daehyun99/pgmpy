import numpy as np
from skbase.base import BaseEstimator as _BaseEstimator

from pgmpy.causal_discovery._base import BaseCausalDiscovery


class BaseParameter(_BaseEstimator):
    """Base class for all parameter classes in pgmpy."""

    _tags = {
        "parameter_type": str,  # classifier, regressor
        "produces_factor": bool,  # Only for TabularCPD
        "is_linear_gaussian": bool,  # Only for LinearGaussianCPD
        "supports_fit_joint": bool,  # Only for FunctionalCPD
        "missing": bool,
        "can_be_root": bool,
    }

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        """Fit parameter to training data.

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        X, y, sample_weight = self._check_fit_data(X, y, sample_weight)
        self._fit(X, y, sample_weight)
        self._is_fitted = True
        return self

    def _fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError

    def predict_proba(self, X=None, n_samples=None):
        """Predict distribution over labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : skpro's Distribution class
        """
        X, _, _ = self._check_predict_proba_data(X)
        y_pred = self._predict_proba(X)
        return y_pred

    def _predict_proba(self, X):
        raise NotImplementedError

    def sample(self, X=None, n_samples=None):
        X, _, _ = self._check_sample_data(X)
        samples = self._sample(X)
        return samples

    def _sample(self, X):
        raise NotImplementedError

    def _check_fit_data(self, X, y=None, sample_weight=None):
        """check train data with tag"""
        transformer = BaseCausalDiscovery()
        X = transformer._check_fit_data(X)

        if y is None:
            self.evidences_ = None
            self.variables_ = transformer.feature_names_in_
        else:
            self.evidences_ = transformer.feature_names_in_
            y = transformer._check_fit_data(y)
            self.variables_ = transformer.feature_names_in_

        # TODO: Implement missing data in suvervised, unsuvervised learning

        # Check sample_weight
        n_samples = len(X)
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)

            if len(sample_weight) != n_samples:
                raise ValueError(f"sample_weight must have length {n_samples}. Got {len(sample_weight)}.")

            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative values.")

        return X, y, sample_weight

    def _check_predict_proba_data(self, X=None, n_samples=None):
        if not self._is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' before calling 'predict_proba'."
            )

    def _check_sample_data(self, X=None, n_samples=None):
        if not self._is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' before calling 'sample'."
            )
        # Supervised
        if (X is None) or (len(X) == len(n_samples)):
            raise ValueError
        else:
            if len(X) == 1:
                # broadcast
                ...

            else:
                ...

        # Unsupervised
        if (X is not None) or (n_samples is None):
            raise ValueError
        else:
            # sampling
            ...
