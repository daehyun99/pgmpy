import numpy as np

from pgmpy.estimators import MaximumLikelihoodEstimator as MLE
from pgmpy.factors.base import BaseFactor
from pgmpy.factors.continuous import LinearGaussianCPD
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
            self._fit_linear()
        elif self.tag_name_ == "functional":
            self._fit_functional()
        elif self.tag_name_ == "skpro":
            self._fit_external_ml()

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

    def _fit_linear(self):
        if self.estimator not in ("MLE", "OLS", None):
            raise ValueError(f"For linear tag, MLE/OLS is supported. Got {self.estimator}")

        target_data = self.data_[self.variable].values

        if not self.parents_:
            mean = np.mean(target_data)
            std = np.std(target_data)
            beta = [mean]
        else:
            evidence_data = self.data_[self.parents_].values

            X = np.c_[np.ones(evidence_data.shape[0]), evidence_data]

            beta, residuals, rank, s = np.linalg.lstsq(X, target_data, rcond=None)
            if len(residuals) > 0:
                variance = residuals[0] / len(target_data)
            else:
                predictions = X @ beta
                variance = np.mean((target_data - predictions) ** 2)
            std = np.sqrt(variance)

        self.fitted_cpd_ = LinearGaussianCPD(variable=self.variable, beta=beta, std=std, evidence=self.parents_)

    def __repr__(self):
        if not getattr(self, "is_fitted_", False):
            tag_display = self.tag[0] if isinstance(self.tag, list) else self.tag
            return (
                f"<FunctionalCPD(variable='{self.variable}', "
                f"tag='{tag_display}', status='unfitted') at {hex(id(self))}>"
            )

        if self.tag_name_ == "tabular" and hasattr(self, "fitted_cpd_"):
            cpd = self.fitted_cpd_
            var_str = f"<FunctionalCPD(tabular) representing P({cpd.variable}:{cpd.variable_card}"

            evidence = cpd.variables[1:]
            evidence_card = cpd.cardinality[1:]

            if evidence:
                evidence_str = " | " + ", ".join([f"{var}:{card}" for var, card in zip(evidence, evidence_card)])
            else:
                evidence_str = ""

            return var_str + evidence_str + f") at {hex(id(self))}>"

        elif self.tag_name_ == "linear" and hasattr(self, "fitted_cpd_"):
            cpd = self.fitted_cpd_
            beta_str = f"{cpd.beta[0]:.3f}"  # 절편 (Intercept)
            for i, parent in enumerate(cpd.evidence):
                beta_str += f" + {cpd.beta[i + 1]:.3f}*{parent}"

            return (
                f"<FunctionalCPD(linear) representing P({cpd.variable} | {', '.join(cpd.evidence)}) "
                f"~ N({beta_str}, std={cpd.std:.3f}) at {hex(id(self))}>"
            )

        return (
            f"<FunctionalCPD(variable='{self.variable}', tag='{self.tag_name_}', status='fitted') at {hex(id(self))}>"
        )
