import numpy as np
from sklearn.linear_model import LinearRegression
from skpro.distfitter.base import BaseDistFitter

from pgmpy.factors.continuous.LinearGaussianCPD import LinearGaussianCPD


class LinearGaussianCPDFitter(BaseDistFitter):
    def __init__(self, estimator="mle", evidences=None):
        self.estimator = estimator
        self.evidences = evidences
        super().__init__()

    def _fit(self, X, y, sample_weight=None):
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

        y_arr = y.to_numpy(dtype=float).reshape(-1)
        if self.evidences is not None:
            self.evidences_ = list(self.evidences)
        else:
            self.evidences_ = list(X.columns)
        if self.estimator == "mle":
            ddof = 0
        elif self.estimator == "unbias":
            ddof = 1 + len(self.evidences_)

        # Supervised Learning
        X_fit = X.loc[:, self.evidences_]

        lm = LinearRegression().fit(X_fit, y_arr, sample_weight)
        residuals = y_arr - lm.predict(X_fit).reshape(-1)
        self.beta_ = np.concatenate(
            [
                np.ravel(lm.intercept_),
                np.ravel(lm.coef_),
            ]
        )
        self.std_ = np.sqrt(np.sum(sample_weight * residuals**2) / (np.sum(sample_weight) - ddof))

        return self

    def _proba(self):
        # intercept = self.beta_[0]
        # coef = self.beta_[1:]
        # mu = intercept + X.loc[:, self.evidences_].to_numpy() @ coef
        # mu = np.asarray(mu, dtype=float).reshape(-1, 1)
        # return Normal(mu=mu, sigma=self.std_, index=X.index)
        return LinearGaussianCPD(...)
