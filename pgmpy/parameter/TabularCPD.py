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

from pgmpy.parameter_estimator.temp_mle import TempMLE

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

    def _fit(self, X, y, sample_weight=None):
        if not hasattr(self, "classes_"):
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(y)
            self.state_names_ = self._label_binarizer.classes_

        estimator_cls = _ESTIMATOR_REGISTRY[self.estimator.lower()]
        self.estimator_ = estimator_cls()
        self.estimator_.fit(
            X,
            y,
            sample_weight=sample_weight,
        )
        self.values_ = np.asarray(self.estimator_.values_)
        self.index_ = pd.RangeIndex(len(self.values_))
        self.columns_ = ["variable"]
        self.is_fitted_ = True

        return self

    def _predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("This TabularCPD instance is not fitted yet. Call 'fit' before calling 'predict_proba'.")

        return CategoricalDistribution(
            values=self.values_,
            state_names=self.state_names_,
            index=self.index_,
            columns=self.columns_,
        )  # (len(X), variable_card)

    def set_values(
        self,
        values,
        evidence_card=None,
        state_names=None,
        parent_order=None,
    ):
        self.values_ = ...
        self.state_names_ = ...
        self.columns_ = ...
        self.is_fitted_ = True
        return self

    def get_values(self):
        result = {
            "values": self.values_,
            "state_names": self.state_names_,
            "index": self.index_,
            "columns": self.columns_,
        }
        return result
