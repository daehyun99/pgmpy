from pgmpy.estimators import MaximumLikelihoodEstimator as MLE
from pgmpy.factors.discrete import TabularCPD


class TabularAdapter:
    """
    Adapter that fits data into a `TabularCPD`.
    """

    def __init__(self, variable, estimator, parents=None):
        self.variable = variable
        self.estimator = estimator
        self.parents = parents if parents is not None else []

    def fit(self, data):
        if self.estimator not in ("MLE", MLE):
            raise ValueError("For tabular tag, only MLE estimator is currently supported.")

        variable_states = sorted(data[self.variable].dropna().unique())
        if not self.parents:
            counts = data[self.variable].value_counts().reindex(variable_states, fill_value=0)
            probs = counts.values / counts.values.sum()
            values = [[prob] for prob in probs]
            self.fitted_cpd_ = TabularCPD(variable=self.variable, variable_card=len(variable_states), values=values)
            return self

        parent_states = [sorted(data[parent].dropna().unique()) for parent in self.parents]
        grouped = (
            data.groupby(self.parents + [self.variable], dropna=False)
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
            evidence=self.parents,
            evidence_card=[len(states) for states in parent_states],
        )
        return self

    def __repr__(self):
        if not hasattr(self, "fitted_cpd_"):
            return f"<TabularAdapter(variable='{self.variable}', status='unfitted') at {hex(id(self))}>"

        cpd = self.fitted_cpd_
        var_str = f"<TabularAdapter representing P({cpd.variable}:{cpd.variable_card}"

        evidence = cpd.variables[1:]
        evidence_card = cpd.cardinality[1:]

        if evidence:
            evidence_str = " | " + ", ".join([f"{var}:{card}" for var, card in zip(evidence, evidence_card)])
        else:
            evidence_str = ""

        return var_str + evidence_str + f") at {hex(id(self))}>"
