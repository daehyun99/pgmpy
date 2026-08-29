from skpro.regression.base import BaseProbaRegressor

class BaseParameter(BaseProbaRegressor):
    """Base class for all parameter classes in pgmpy."""

    _tags = {
        "object_type": set,  # {"continuous", "discrete", "mixture"}
        "fit_mode": set,  # {"supervise", "unsupervise", "untraninable"}
        "python_dependencies": [],
        "local:plug_in": set,
        "global:plug_in": set,
        "local:full_bayesian": set,
        "global:full_bayesian": set,
    }

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, sample_weight=None):
        """Fit parameters to training data.

        If a parameter corresponds to a root node of the Bayesian network,
        only unsupervised learning is supported.
        If a parameter does not correspond to a root node of the Bayesian network,
        only supervised learning is supported.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data for supervised or unsupervised learning.

        y : pandas.DataFrame, (default=None)
            Target labels for supervised learning.
            Must be None for unsupervised learning.

        sample_weight : numpy.ndarray, (default=None)
            Sample weights. If None, all samples are given equal weight.

        Returns
        -------
        self : pgmpy parameter class
            Returns the parameter instance with the fitted attributes.
        """
        if hasattr(self, "object_type_") is False:
            raise AttributeError(f"{self.__class__.__name__}._fit() must set the 'object_type_' attribute.")
        if hasattr(self, "fit_mode_") is False:
            raise AttributeError(f"{self.__class__.__name__}._fit() must set the 'fit_mode_' attribute.")

        self._is_fitted = True
        return self

    def sample(self, X=None, n_samples=None):
        """Draw samples from the probability distribution or conditional probability distribution.

        If the parameter was fitted using supervised learning, samples are drawn from
        the conditional probability distribution given X.
        If the parameter was fitted using unsupervised learning, samples are drawn from
        the (unconditional) probability distribution.

        Parameters
        ----------
        X : pandas.DataFrame, (default=None)
            Required when the parameter was fitted with supervised learning.
            Raises an error if the parameter was fitted with unsupervised learning;
            use ``n_samples`` instead.

        n_samples : int, (default=None)
            Required when the parameter was fitted with unsupervised learning.
            Raises an error if the parameter was fitted with supervised learning;
            use ``X`` instead.

        Returns
        -------
        y : numpy.ndarray or torch.Tensor
            Samples drawn from the distribution.

        log_proba : numpy.ndarray or torch.Tensor
            Log pdf or log pmf values corresponding to the drawn samples.

        See Also
        --------
        predict_proba : Return the predicted conditional probability distribution.
        """
        X, n_samples = self._check_sample_data(X, n_samples)
        samples, log_proba = self._sample(X, n_samples)
        return samples, log_proba

    def _sample(self, X, n_samples):
        raise NotImplementedError
