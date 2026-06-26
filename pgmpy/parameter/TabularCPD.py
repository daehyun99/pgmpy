import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from pgmpy.distributions.categorical import CategoricalDistribution
from pgmpy.parameter._base import BaseParameter


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
        prior_type=None,
        equivalent_sample_size=None,
        pseudo_counts=None,
    ):
        self.prior_type = prior_type
        self.equivalent_sample_size = equivalent_sample_size
        self.pseudo_counts = pseudo_counts
        self.is_fitted_ = False
        super().__init__()

    def _fit(self, X, y=None, sample_weight=None):
        if not hasattr(self, "categories_"):
            self._label_binarizer = LabelBinarizer()
            if y is None:  # if root node
                self._label_binarizer.fit(X)
            else:
                self._label_binarizer.fit(y)
            self.categories_ = self._label_binarizer.classes_

        X = X.loc[:, sorted(X.columns)].copy()

        feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )

        if y is None:
            # Unsupervised Learning: Root node
            feature_names = feature_names_in_.tolist()

            counts = X.groupby(
                feature_names,
                observed=True,
                sort=True,
            ).size()

            if self.categories_ is not None:
                counts = counts.reindex(
                    self.categories_,
                    fill_value=0,
                )

            self.CPT_ = counts.div(counts.sum()).to_frame(name=0)

            self.evidence_names_ = np.asarray(
                [],
                dtype=object,
            )
            self.evidence_states_ = self.CPT_.columns

        else:
            # Supervised Learning
            target_name = "__target__"

            while target_name in X.columns:
                target_name = f"_{target_name}"

            X[target_name] = np.asarray(y)

            evidence_names = feature_names_in_.tolist()

            counts = (
                X.groupby(
                    [target_name, *evidence_names],
                    observed=True,
                    sort=True,
                    dropna=False,
                )
                .size()
                .unstack(
                    evidence_names,
                    fill_value=0,
                )
                .reindex(
                    index=self.categories_,
                    fill_value=0,
                )
                .rename_axis(index=None)
            )

            self.CPT_ = counts.div(
                counts.sum(axis=0),
                axis=1,
            )

            self.evidence_names_ = np.asarray(
                evidence_names,
                dtype=object,
            )
            self.evidence_states_ = self.CPT_.columns

        self.CPT_ = np.asarray(self.CPT_)

        self.columns_ = [y.name if getattr(y, "name", None) is not None else "variable"]

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
        column_positions = self.evidence_states_.get_indexer(row_evidence)
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
