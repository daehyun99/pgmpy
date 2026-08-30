import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from skpro.distfitter.base import BaseDistFitter

from pgmpy.factors.discrete.CPD import TabularCPD


class TabularCPDfitter(BaseDistFitter):
    def __init__(
        self,
        categories=None,
        evidences=None,
    ):
        self.categories = categories
        self.evidences = evidences
        super().__init__()

    def _fit(self, X, y, sample_weight=None):
        # Check sample_weight
        n_samples = len(X)
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)

            if len(sample_weight) != n_samples:
                raise ValueError(f"sample_weight must have length {n_samples}. Got {len(sample_weight)}.")

            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative values.")

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

        # Supervised Learning
        df = pd.concat([X, y], axis=1)

        evidence_names = list(X.columns)

        weights = pd.Series(
            sample_weight,
            index=df.index,
        )

        counts = weights.groupby(
            [df[column] for column in [y.columns[0], *evidence_names]],
            observed=True,
            sort=True,
            dropna=False,
        ).sum()

        counts = (
            counts.unstack(
                evidence_names,
                fill_value=0,
            )
            .reindex(
                index=self.categories_[y.columns[0]],
                fill_value=0,
            )
            .rename_axis(index=None)
        )

        self.CPT_ = counts.div(
            counts.sum(axis=0),
            axis=1,
        )

        self.CPT_ = np.asarray(self.CPT_)
        return self

    def _proba(self):

        # row_evidence = pd.MultiIndex.from_frame(X.loc[:, self.evidences_.keys()])
        # cpt_column_index = pd.MultiIndex.from_product(
        #     [self.evidences_[name] for name in list(self.evidences_.keys())],
        #     names=list(self.evidences_.keys()),
        # )
        # column_positions = cpt_column_index.get_indexer(row_evidence)
        # probabilities = self.CPT_[:, column_positions].T

        # return NominalDistribution(
        #     probs=probabilities,
        #     categories=self.categories_[next(iter(self.categories_))],
        #     columns=[next(iter(self.categories_))],
        # )  # (len(X), variable_card)
        return TabularCPD(...)
