from pgmpy.estimators import MaximumLikelihoodEstimator as MLE
from pgmpy.factors.base import BaseFactor


class FunctionalCPD(BaseFactor):
    def __init__(self, variable, tag, estimator):
        self.variable = variable
        self.tag = tag
        self.estimator = estimator

    def fit(self, data):
        self.parents_ = list(self.predecessors(self.variable))
        self.tag_name_ = self.tag[0] if isinstance(self.tag, list) else self.tag

        if self.tag_name_ == "tabular":
            self._fit_tabular()
        elif self.tag_name_ == "linear":
            self._fit_linear(data)
        elif self.tag_name_ == "functional":
            self._fit_functional(data)
        elif self.tag_name_ == "skpro":
            self._fit_external_ml(data)

        self.is_fitted_ = True

        return self

    def _fit_tabular(self):
        self.estimator

        if isinstance(self.estimator, MLE):
            self.estimator.estimate_cpd(self.variable)
