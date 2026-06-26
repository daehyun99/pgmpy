import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

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
        # variable_card=None,
        # evidence_card=None,
        categories=None,
        evidences=None,
        prior_type=None,
        equivalent_sample_size=None,
        pseudo_counts=None,
    ):
        # self.variable_card = variable_card
        # self.evidence_card = evidence_card
        self.categories = categories
        self.evidences = evidences
        self.prior_type = prior_type
        self.equivalent_sample_size = equivalent_sample_size
        self.pseudo_counts = pseudo_counts
        self._is_fitted = False
        super().__init__()

    def _fit(self, X, y=None, sample_weight=None):
        if y is None:
            # Unsupervised Learning
            if self.categories is None:
                self._y_transformer = LabelBinarizer()
                self._y_transformer.fit(X)
                self.categories_ = {X.columns[0]: self._y_transformer.classes_}
                self.evidences_ = self.evidences
            else:
                self.categories_ = self.categories
                self.evidences_ = self.evidences
        else:
            # Supervised Learning
            if self.categories is None:
                self._y_transformer = LabelBinarizer()
                self._y_transformer.fit(y)
                self.categories_ = {y.columns[0]: self._y_transformer.classes_}
            else:
                self.categories_ = self.categories

            if self.evidences is None:
                self._X_transformer = OneHotEncoder(
                    categories="auto",
                    handle_unknown="ignore",
                )
                self._X_transformer.fit(X)
                self.evidences_ = {
                    column: categories.tolist()
                    for column, categories in zip(
                        X.columns,
                        self._X_transformer.categories_,
                    )
                }
            else:
                self.evidences_ = self.evidences

        if y is None:
            # Unsupervised Learning: Root node
            counts = X.groupby(
                list(X.columns),
                observed=True,
                sort=True,
            ).size()

            counts = counts.reindex(
                self.categories_[X.columns[0]],
                fill_value=0,
            )

            self.columns_ = [X.columns[0]]
            self.CPT_ = counts.div(counts.sum()).to_frame(name="prob")

        else:
            # Supervised Learning
            df = pd.concat([X, y], axis=1)

            evidence_names = list(X.columns)

            counts = (
                df.groupby(
                    [y.columns[0], *evidence_names],
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
                    index=self.categories_[y.columns[0]],
                    fill_value=0,
                )
                .rename_axis(index=None)
            )

            self.columns_ = [y.columns[0]]
            self.CPT_ = counts.div(
                counts.sum(axis=0),
                axis=1,
            )

        self.CPT_ = np.asarray(self.CPT_)
        self._is_fitted = True

        return self

    def _predict_proba(self, X):

        if not self._is_fitted:
            raise RuntimeError("This TabularCPD instance is not fitted yet. Call 'fit' before calling 'predict_proba'.")

        if self.evidences_ is None:
            # Unsupervised Learning
            probabilities = np.repeat(
                np.asarray(self.CPT_).T,
                repeats=len(X),
                axis=0,
            )

            return CategoricalDistribution(
                probs=probabilities,
                categories=self.categories_[self.columns_[0]],
                columns=self.columns_,
            )

        row_evidence = pd.MultiIndex.from_frame(X.loc[:, self.evidences_.keys()])
        cpt_column_index = pd.MultiIndex.from_product(
            [self.evidences_[name] for name in list(self.evidences_.keys())],
            names=list(self.evidences_.keys()),
        )
        column_positions = cpt_column_index.get_indexer(row_evidence)
        probabilities = self.CPT_[:, column_positions].T

        return CategoricalDistribution(
            probs=probabilities,
            categories=self.categories_[self.columns_[0]],
            columns=self.columns_,
        )  # (len(X), variable_card)

    def set_values(self, CPT, columns, categories, evidence_states, evidence_names, is_fitted):
        self.CPT_ = CPT
        self.columns_ = columns
        self.categories_ = categories
        self.evidence_states_ = evidence_states
        self.evidence_names_ = evidence_names
        self._is_fitted = is_fitted
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
