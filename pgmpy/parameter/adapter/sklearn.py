from sklearn.base import is_classifier, is_regressor
from skpro.regression.base._delegate import _DelegatedProbaRegressor

from .._base import BaseParameter


class SklearnAdapter(_DelegatedProbaRegressor, BaseParameter):
    _tags = {
        "produces_factor": False,
        "is_linear_gaussian": False,
        "missing": False,
        "supports_fit_joint": False,
        "can_be_root": False,
        "python_dependencies": (),
    }

    def __init__(self, estimator):
        self.estimator = estimator
        self.estimator_ = estimator
        super().__init__()

        if is_regressor(estimator):
            self.set_tags(variable_type="continuous")
        elif is_classifier(estimator):
            self.set_tags(variable_type="discrete")

    def _fit(self, X, y, sample_weight=None):

        estimator = self._get_delegate()
        estimator.fit(X=X, y=y, sample_weight=sample_weight)
        return self

    def _predict_proba(self, X):
        estimator = self._get_delegate()

        if is_regressor(estimator):
            from skpro.distributions.normal import Normal

            mu = estimator.predict(X)
            sigma = 1.0  # TODO: Change this value
            return Normal(mu, sigma)

        elif is_classifier(estimator):
            from pgmpy.distributions.nominal import NominalDistribution

            return NominalDistribution(probs=[0.1, 0.2], categories=["1", "2"])  # TODO: Change this value
