from skpro.regression.base._delegate import _DelegatedProbaRegressor

from pgmpy.parameter._base import BaseParameter


class SkproAdapter(_DelegatedProbaRegressor, BaseParameter):
    _tags = {
        "variable_type": "continuous",
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

    def _fit(self, X, y, sample_weight=None):

        estimator = self._get_delegate()
        estimator.fit(X=X, y=y, C=sample_weight)
        return self

    def _predict_proba(self, X):
        estimator = self._get_delegate()
        return estimator.predict_proba(X=X)
