from skbase.base import BaseEstimator as _BaseEstimator


class BaseParameter(_BaseEstimator):
    """Base class for all parameter classes in pgmpy."""

    _config = {}

    _tags = {
        "variable_type": "discrete",
        "produces_factor": False,
        "is_linear_gaussian": False,
        "supports_fit_joint": False,
        "python_dependencies": (),
    }

    @property
    def name(self):
        """Return the name of the object or estimator."""
        return self.__class__.__name__
