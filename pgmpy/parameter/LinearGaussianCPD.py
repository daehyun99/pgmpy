from skpro.distributions.normal import Normal

from pgmpy.parameter._base import BaseParameter


class LinearGaussianCPD(BaseParameter):
    """LinearGaussianCPD"""

    _tags = {
        "variable_type": "continues",
        "produces_factor": False,
        "is_linear_gaussian": True,
        "missing": False,
        "supports_fit_joint": False,
        "can_be_root": True,
        "python_dependencies": (),
    }

    def __init__(self): ...

    def _fit(self, X, y=None, sample_weight=None): ...

    def _predict_proba(self, X):
        ...
        return Normal(mu, sigma)

    def set_values(self, beta, std, is_fitted): ...
