from skbase.base import BaseEstimator as _BaseEstimator


class BaseParameter(_BaseEstimator):
    """Base class for all parameter classes in pgmpy."""

    _tags = {
        "variable_type": str,
        "produces_factor": bool,
        "is_linear_gaussian": bool,
        "supports_fit_joint": bool,
        "python_dependencies": tuple,
    }

    def fit(self, X, y):
        """API docs"""
        return self._fit(X, y)

    def _fit(self, X, y):
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
