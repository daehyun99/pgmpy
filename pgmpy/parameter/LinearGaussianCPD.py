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

    def _predict_proba(self, X):
        ...
        return Normal(mu, sigma)

    def set_values(self, beta, std, is_fitted): ...
