import numpy as np
from sklearn.linear_model import LinearRegression
from skpro.distributions.normal import Normal

from pgmpy.parameter._base import BaseParameter


class LinearGaussianCPD(BaseParameter):
    """LinearGaussianCPD"""

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": True,
        "missing": False,
        "supports_fit_joint": False,
        "can_be_root": True,
        "python_dependencies": (),
    }

    def __init__(self, estimator="mle"):
        self.estimator = estimator

    def _fit(self, X, y=None, sample_weight=None):
        if y is None:
            x_arr = X.to_numpy(dtype=float).reshape(-1)
            y_arr = None
            if self.estimator == "mle":
                ddof = 0
            elif self.estimator == "unbias":
                ddof = 1
        else:
            x_arr = X.to_numpy(dtype=float).reshape(-1)
            y_arr = y.to_numpy(dtype=float).reshape(-1)
            if self.estimator == "mle":
                ddof = 0
            elif self.estimator == "unbias":
                ddof = 1 + X.shape[1]

        if y_arr is None:
            # Unsupervised Learning
            w_mean = np.sum(sample_weight * x_arr) / np.sum(sample_weight)
            w_var = np.sum(sample_weight * (x_arr - w_mean) ** 2) / (np.sum(sample_weight) - ddof)

            self.beta_ = [w_mean]
            self.std_ = np.sqrt(w_var)
        else:
            # Supervised Learning
            lm = LinearRegression().fit(X, y_arr, sample_weight)
            residuals = y_arr - lm.predict(X).reshape(-1)
            self.beta_ = np.concatenate(
                [
                    np.ravel(lm.intercept_),
                    np.ravel(lm.coef_),
                ]
            )
            self.std_ = np.sqrt(np.sum(sample_weight * residuals**2) / (np.sum(sample_weight) - ddof))
        return self

    def _predict_proba(self, X):
        n = len(X)
        if len(self.beta_) == 1:
            # Unsupervised
            mu = np.full(n, self.beta_[0], dtype=float)
            mu = np.asarray(mu, dtype=float).reshape(-1, 1)
        else:
            # Supervised
            intercept = self.beta_[0]
            coef = self.beta_[1:]
            mu = intercept + X.to_numpy() @ coef
            mu = np.asarray(mu, dtype=float).reshape(-1, 1)
        return Normal(mu=mu, sigma=self.std_, index=X.index)

    def set_fitted_params(self, beta, std, is_fitted):
        self.beta_ = beta
        self.std_ = std
        self._is_fitted = is_fitted
        return self
