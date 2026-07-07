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
            # Unsupervised Learning
            if sample_weight is None:
                ddof = 0 if self.estimator == "mle" else 1
                self.beta_ = [X.mean()]
                self.std_ = X.std(ddof=ddof)
        else:
            # Supervised Learning
            lm = LinearRegression().fit(X, y)
            residuals = y.to_numpy() - lm.predict(X)
            ddof = 0 if self.estimator == "mle" else 1 + X.shape[1]
            self.beta_ = np.concatenate(
                [
                    np.ravel(lm.intercept_),
                    np.ravel(lm.coef_),
                ]
            )
            self.std_ = np.sqrt(np.sum(residuals**2) / (len(residuals) - ddof))
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

    def set_values(self, beta, std, is_fitted): ...
