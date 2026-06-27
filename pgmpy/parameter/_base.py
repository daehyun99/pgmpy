from skbase.base import BaseEstimator as _BaseEstimator


class BaseParameter(_BaseEstimator):
    """Base class for all parameter classes in pgmpy."""

    _tags = {
        "variable_type": str,
        "produces_factor": bool,
        "is_linear_gaussian": bool,
        "missing": bool,
        "supports_fit_joint": bool,
        "can_be_root": bool,
        "python_dependencies": tuple,
    }

    def fit(self, X, y=None, sample_weight=None):
        """API docs"""
        self._check_data(X, y, sample_weight)
        return self._fit(X, y, sample_weight)

    def _fit(self, X, y, sample_weight):
        """Fit parameter to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """API docs"""
        self._check_data(X)
        y_pred = self._predict_proba(X)
        return y_pred

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        raise NotImplementedError

    def _check_data(self, X, y=None, sample_weight=None):
        """check train data with tag"""
        # TODO: Implement when #3455
        # Is this data pd.DataFrame?
        # Is this produces factor?
        # Is data format match with tag(`variable_type`)?
        pass

    def _check_is_fitted(self):
        pass
