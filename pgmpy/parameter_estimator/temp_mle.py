import numpy as np

class TempMLE:
    _tags = {
        "allow_variable_type": "discrete",
    }

    def __init__(self):
        pass

    def fit(self, X, y, sample_weight=None):
        X = X.loc[:, sorted(X.columns)].copy()

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.state_names_ = np.unique(np.asarray(y))

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
                index=self.state_names_,
                fill_value=0,
            )
            .rename_axis(index=None)
        )

        self.values_ = counts.div(
            counts.sum(axis=0),
            axis=1,
        )

        self.evidence_names_ = np.asarray(evidence_names, dtype=object)
        self.evidence_states_ = self.values_.columns

        return self
