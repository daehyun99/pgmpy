import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from skpro.distfitter.base import BaseDistFitter


# todo: change class name and write docstring
class Nominalfitter(BaseDistFitter):
    def __init__(self, dist):

        self.dist = dist

        super().__init__()

    def __dynamic_tags__(self):
        pass

    def __post_init__(self):

        pass

    def _fit(self, X, C=None):
        """Fit distribution to data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            Data to fit the distribution to.
        C : pandas DataFrame, optional (default=None)
            Censoring indicator for survival analysis.
            Only passed if ``capability:survival`` tag is True.

        Returns
        -------
        self : reference to self
        """
        # Check sample_weight
        n_samples = len(X)
        sample_weight = np.ones(n_samples, dtype=float)

        # Unsupervised Learning
        if self.categories is None:
            self._y_transformer = LabelBinarizer()
            self._y_transformer.fit(X)
            self.categories_ = {X.columns[0]: self._y_transformer.classes_}
            self.evidences_ = self.evidences
        else:
            self.categories_ = self.categories
            self.evidences_ = self.evidences

        # Unsupervised Learning: Root node
        weights = pd.Series(
            sample_weight,
            index=X.index,
        )
        counts = weights.groupby(
            [X[column] for column in list(X.columns)],
            observed=True,
            sort=True,
        ).sum()

        counts = counts.reindex(
            self.categories_[X.columns[0]],
            fill_value=0,
        )

        self.CPT_ = counts.div(counts.sum()).to_frame(name="prob")
        return self

    # todo: implement this, mandatory
    def _proba(self):
        """Return fitted scalar distribution.

        State required:
            Requires state to be "fitted" = self.is_fitted=True

        Accesses in self:
            Fitted model attributes ending in "_"

        Returns
        -------
        dist : skpro BaseDistribution (scalar)
            Distribution fitted to data passed in ``fit``.
        """
        from pgmpy.distributions.nominal import NominalDistribution

        # Unsupervised Learning
        probabilities = np.repeat(
            np.asarray(self.CPT_).T,
            repeats=1,
            axis=0,
        )

        return NominalDistribution(
            probs=probabilities,
            categories=self.categories_[next(iter(self.categories_))],
            columns=[next(iter(self.categories_))],
        )
