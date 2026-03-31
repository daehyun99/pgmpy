from pgmpy.estimators import MaximumLikelihoodEstimator as MLE
from pgmpy.factors.base import BaseFactor
from pgmpy.factors.discrete import TabularCPD


class FunctionalCPD(BaseFactor):
    def __init__(self, variable, tag, estimator):
        self.variable = variable
        self.tag = tag
        self.estimator = estimator

    def fit(self, data, target=None, parents=None):
        self.data_ = data
        self.parents_ = parents if parents is not None else []
        if target is not None:
            self.variable = target
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
        if self.estimator not in ("MLE", MLE):
            raise ValueError("For tabular tag, only MLE estimator is currently supported.")

        variable_states = sorted(self.data_[self.variable].dropna().unique())
        if not self.parents_:
            counts = self.data_[self.variable].value_counts().reindex(variable_states, fill_value=0)
            probs = counts.values / counts.values.sum()
            values = [[prob] for prob in probs]
            self.fitted_cpd_ = TabularCPD(variable=self.variable, variable_card=len(variable_states), values=values)
            return

        parent_states = [sorted(self.data_[parent].dropna().unique()) for parent in self.parents_]
        grouped = (
            self.data_.groupby(self.parents_ + [self.variable], dropna=False)
            .size()
            .unstack(self.variable, fill_value=0)
            .reindex(columns=variable_states, fill_value=0)
        )
        grouped = grouped.T
        grouped = grouped / grouped.sum(axis=0).replace(0, 1)
        self.fitted_cpd_ = TabularCPD(
            variable=self.variable,
            variable_card=len(variable_states),
            values=grouped.values,
            evidence=self.parents_,
            evidence_card=[len(states) for states in parent_states],
        )
