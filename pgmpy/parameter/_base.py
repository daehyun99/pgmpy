import numpy as np
import pandas as pd
from skbase.base import BaseEstimator as _BaseEstimator

from pgmpy.causal_discovery._base import BaseCausalDiscovery


class BaseParameter(_BaseEstimator):
    """Base class for all parameter classes in pgmpy."""

    _tags = {
        "object_type": str,  # {"continuous", "discrete", "mixture"}
        "fit_mode": set,  # {"supervise", "unsupervise", "untraninable"}
        "python_dependencies": set,
        "local:plug_in": set,
        "global:plug_in": set,
        "local:full_bayesian": set,
        "global:full_bayesian": set,
    }

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        """Fit parameter to training data.

        Parameters
        ----------
        X : pandas DataFrame

        y : pandas DataFrame

        Returns
        -------
        self : reference to self
        """
        X, y, sample_weight = self._check_fit_data(X, y, sample_weight)
        self._fit(X, y, sample_weight)

        if hasattr(self, "object_type_") is False:
            raise AttributeError(f"{self.__class__.__name__}._fit() must set the 'object_type_' attribute.")
        if hasattr(self, "fit_mode_") is False:
            raise AttributeError(f"{self.__class__.__name__}._fit() must set the 'fit_mode_' attribute.")

        self._is_fitted = True
        return self

    def _fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError

    def predict_proba(self, X):
        """Predict distribution over labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`

        Returns
        -------
        y : skpro's Distribution class
        """
        X = self._check_predict_proba_data(X)
        y_pred = self._predict_proba(X)
        return y_pred

    def _predict_proba(self, X):
        raise NotImplementedError

    def sample(self, X=None, n_samples=None):
        """Sample labels from the predicted distribution for data from features.

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`

        n_samples : int

        Returns
        -------
        y : pandas DataFrame
            samples drawn from the predicted distribution
        """
        X, n_samples = self._check_sample_data(X, n_samples)
        samples = self._sample(X, n_samples)
        return samples

    def _sample(self, X, n_samples):
        raise NotImplementedError

    def _check_fit_data(self, X, y=None, sample_weight=None):
        """check train data with tag"""
        if (y is None) and ("unsupervise" in self.get_class_tag("fit_mode")):
            transformer = BaseCausalDiscovery()
            X = transformer._check_fit_data(X)
            self.fit_mode_ = "unsupervise"
            self.evidences_ = None
            self.variables_ = transformer.feature_names_in_
            if (self.get_class_tag("object_type") == "continuous") and (
                pd.api.types.is_numeric_dtype(X[self.variables_[0]])
            ):
                self.object_type_ = "continuous"
            elif (self.get_class_tag("object_type") == "discrete") and (
                pd.api.types.is_string_dtype(X[self.variables_[0]])
            ):
                self.object_type_ = "discrete"
            else:
                raise ValueError

        elif (y is not None) and ("supervise" in self.get_class_tag("fit_mode")):
            transformer = BaseCausalDiscovery()
            X = transformer._check_fit_data(X)
            self.fit_mode_ = "supervise"
            self.evidences_ = transformer.feature_names_in_
            y = transformer._check_fit_data(y)
            self.variables_ = transformer.feature_names_in_
            if ("continuous" in self.get_class_tag("object_type")) and (
                pd.api.types.is_numeric_dtype(y[self.variables_[0]])
            ):
                self.object_type_ = "continuous"
            elif ("discrete" in self.get_class_tag("object_type")) and (
                pd.api.types.is_string_dtype(y[self.variables_[0]])
            ):
                self.object_type_ = "discrete"
            else:
                raise ValueError
        else:
            raise ValueError

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

    def _check_predict_proba_data(self, X):
        if not self._is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' before calling 'predict_proba'."
            )
        if self.fit_mode_ in {"unsupervise", "untrainable"}:
            raise NotImplementedError

        transformer = BaseCausalDiscovery()
        X = transformer._check_fit_data(X)
        return X

    def _check_sample_data(self, X=None, n_samples=None):
        if not self._is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' before calling 'sample'."
            )
        if (X is None) and (n_samples is None):
            raise ValueError

        if self.fit_mode_ in {"unsupervise", "untrainable"}:
            if X is not None:
                raise ValueError

        elif self.fit_mode_ == "supervise":
            if X is None:
                raise ValueError

        return X, n_samples
