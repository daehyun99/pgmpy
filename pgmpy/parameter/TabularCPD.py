import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from pgmpy.distributions.categorical import CategoricalDistribution
from pgmpy.parameter._base import BaseParameter
from pgmpy.parameter_estimator import (
    DiscreteBayesianEstimator,
    DiscreteEM,
    DiscreteMLE,
)
from pgmpy.parameter_estimator.temp_mle import TempMLE  # DiscreteMLE

_ESTIMATOR_REGISTRY = {
    "mle": DiscreteMLE,
    "bayesian": DiscreteBayesianEstimator,
    "em": DiscreteEM,
    "temp": TempMLE,
}


class TabularCPD(BaseParameter):
    """TabularCPD"""

    _tags = {
        "variable_type": "discrete",
        "produces_factor": True,
        "is_linear_gaussian": False,
        "missing": False,
        "supports_fit_joint": False,
        "python_dependencies": ("skpro"),
    }

    def __init__(
        self,
        estimator="temp",
        prior_type=None,
        equivalent_sample_size=10,
        pseudo_counts=None,
    ):
        self.estimator = estimator
        self.prior_type = prior_type
        self.equivalent_sample_size = equivalent_sample_size
        self.pseudo_counts = pseudo_counts
        self.is_fitted_ = False
        super().__init__()

    def _fit(self, X, y=None, sample_weight=None):
        if not hasattr(self, "state_names_"):
            self._label_binarizer = LabelBinarizer()
            if y is None:  # if root node
                self._label_binarizer.fit(X)
            else:
                self._label_binarizer.fit(y)
            self.categories_ = self._label_binarizer.classes_

        estimator_cls = _ESTIMATOR_REGISTRY[self.estimator.lower()]
        self.estimator_ = estimator_cls()
        self.estimator_.fit(
            X,
            y,
            sample_weight,
            self.categories_,
            prior_type=self.prior_type,
            equivalent_sample_size=self.equivalent_sample_size,
            pseudo_counts=self.pseudo_counts,
        )
        self.CPT_ = np.asarray(self.estimator_.CPT_)
        self.columns_ = [y.name if getattr(y, "name", None) is not None else "variable"]
        self.evidence_states_ = self.estimator_.evidence_states_
        self.evidence_names_ = np.asarray(
            self.estimator_.evidence_names_,
            dtype=object,
        )
        self.is_fitted_ = True

        return self

    def _predict_proba(self, X):

        if not self.is_fitted_:
            raise RuntimeError("This TabularCPD instance is not fitted yet. Call 'fit' before calling 'predict_proba'.")

        evidence_names = self.evidence_names_

        if evidence_names.size == 0:
            probabilities = np.repeat(
                np.asarray(self.CPT_).T,
                repeats=len(X),
                axis=0,
            )

            return CategoricalDistribution(
                probs=probabilities,
                categories=self.categories_,
                columns=self.columns_,
            )

        evidence_columns = self.evidence_states_.names
        row_evidence = pd.MultiIndex.from_frame(X.loc[:, evidence_columns])
        column_positions = self.estimator_.evidence_states_.get_indexer(row_evidence)
        probabilities = self.CPT_[:, column_positions].T

        return CategoricalDistribution(
            probs=probabilities,
            categories=self.categories_,
            columns=self.columns_,
        )  # (len(X), variable_card)

    def set_values(self, CPT, columns, categories, evidence_states, evidence_names):
        self.CPT_ = CPT
        self.columns_ = columns
        self.categories_ = categories
        self.evidence_states_ = evidence_states
        self.evidence_names_ = evidence_names
        self.is_fitted_ = True
        return self

    def get_values(self):
        attributes = {
            "CPT": "CPT_",
            "columns": "columns_",
            "categories": "categories_",
            "evidence_states": "evidence_states_",
            "evidence_names": "evidence_names_",
        }

        return {key: getattr(self, attr_name) for key, attr_name in attributes.items() if hasattr(self, attr_name)}
