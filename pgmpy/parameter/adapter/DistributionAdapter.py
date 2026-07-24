from pgmpy.parameter._base import BaseParameter


class DistributionAdapter(BaseParameter):
    def __init__(self, distribution):
        self.distribution = distribution
        ...

    def _fit(X, y):
        raise NotImplementedError

    def _predict_proba(X):
        # n_size = len(X)
        distribution = ...  # broadcast the self.distribution, shapes: (len(X), ...)
        return distribution
