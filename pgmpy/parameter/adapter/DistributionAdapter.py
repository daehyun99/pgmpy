from skpro.distributions.base import BaseDistribution

from pgmpy.parameter._base import BaseParameter


class DistributionAdapter(BaseParameter):
    _tags = {
        "parameter_type": "distribution",
        "produces_factor": False,
        "is_linear_gaussian": False,
        "supports_fit_joint": False,
        "missing": False,
        "can_be_root": True,
    }

    def __init__(self, distribution: BaseDistribution):
        if not isinstance(distribution, BaseDistribution):
            raise TypeError(
                f"distribution must be an instance of a skpro distribution. Received: {type(distribution).__name__}"
            )

        self.distribution = distribution
        super().__init__()
        self._is_fitted = True

    def _fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError

    def _predict_proba(self, X):
        """Broadcast the fixed distribution to the rows of ``X``."""
        distribution_params = self.distribution.get_params(deep=False)
        distribution_params.update(index=X.index)
        return type(self.distribution)(**distribution_params)
