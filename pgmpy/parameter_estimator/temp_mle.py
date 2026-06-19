import numpy as np


class TempMLE:
    _tags = {
        "allow_variable_type": "discrete",
    }

    def __init__(self):
        pass

    def fit(
        self,
        X,
        y=None,
        sample_weight=None,
        categories=None,
        prior_type=None,
        equivalent_sample_size=10,
        pseudo_counts=None,
    ):
        X = X.loc[:, sorted(X.columns)].copy()

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        if y is None:
            feature_names = self.feature_names_in_.tolist()

            counts = X.groupby(
                feature_names,
                observed=True,
                sort=True,
            ).size()

            if categories is not None:
                counts = counts.reindex(categories, fill_value=0)

            self.CPT_ = counts.div(counts.sum()).to_frame(name=0)

            self.evidence_names_ = np.asarray([], dtype=object)
            self.evidence_states_ = self.CPT_.columns

            return self
        target_name = "__target__"
        while target_name in X.columns:
            target_name = f"_{target_name}"

        X[target_name] = np.asarray(y)

        evidence_names = self.feature_names_in_.tolist()

        counts = (
            X.groupby(
                [target_name, *evidence_names],
                observed=True,
                sort=True,
                dropna=False,
            )
            .size()
            .unstack(evidence_names, fill_value=0)
            .reindex(
                index=categories,
                fill_value=0,
            )
            .rename_axis(index=None)
        )

        self.CPT_ = counts.div(
            counts.sum(axis=0),
            axis=1,
        )

        self.evidence_names_ = np.asarray(evidence_names, dtype=object)
        self.evidence_states_ = self.CPT_.columns

        return self
