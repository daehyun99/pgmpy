from skbase.utils.dependencies import _safe_import

from pgmpy.parameter.bayesian._base import BasePyroRegression

torch = _safe_import("torch")
pyro = _safe_import("pyro", pkg_name="pyro-ppl")


class BayesianMCMCRegression(BasePyroRegression):
    """
    Bayesian Functional Regression class with MCMC sampling.

    Parameters
    ----------
    model : callable
    warmup_steps : int, optional
    num_samples : int, optional
    posterior : object, optional
    return_sites : str, optional
    device : str, optional

    Example
    -------
    >>> from pgmpy.parameter.bayesian.BayesianMCMCRegression import BayesianMCMCRegression
    >>> from skbase.utils.dependencies import _safe_import
    >>> import numpy as np
    >>> import pandas as pd
    >>> torch = _safe_import("torch")
    >>> pyro = _safe_import("pyro", pkg_name="pyro-ppl")
    >>> rng = np.random.default_rng(42)
    >>> n_samples = 10_000

    >>> A = rng.normal(
    ...     loc=0.0,
    ...     scale=1.0,
    ...     size=n_samples,
    ... )

    >>> B = (
    ...     A + rng.normal(
    ...         loc=1.0,
    ...         scale=0.5,
    ...         size=n_samples,
    ...     )
    ... )

    >>> X = pd.DataFrame(
    ...     {
    ...         "A": A,
    ...     }
    ... )
    >>> y = pd.DataFrame(
    ...     {
    ...         "B": B,
    ...     }
    ... )

    >>> def model(X_tensor, y_tensor=None):
    ...     intercept = pyro.sample("intercept", pyro.distributions.Normal(0.0, 10.0))
    ...     coeff = pyro.sample("coeff", pyro.distributions.Normal(0.0, 1.0))
    ...     sigma = pyro.sample("sigma", pyro.distributions.Uniform(0.0, 10.0))
    ...     mean = intercept + coeff * X_tensor[:, 0]
    ...     with pyro.plate("data", len(X_tensor[:, 0])):
    ...         pyro.sample("obs", pyro.distributions.Normal(mean, sigma), obs=y_tensor)

    >>> regressor = BayesianMCMCRegression(
    ...     model=model,
    ...     warmup_steps=10,
    ...     num_samples=100,
    ... )
    >>> regressor.fit(X, y) # doctest: +SKIP
    BayesianMCMCRegression()
    >>> normal = regressor.predict_proba(X[:5]) # doctest: +SKIP
    >>> normal # doctest: +SKIP
    Normal()

    """

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": False,
        "missing": False,
        "supports_fit_joint": False,
        "can_be_root": False,
        "python_dependencies": ("pyro-ppl"),
    }

    def __init__(
        self,
        model,
        warmup_steps=200,
        num_samples=1000,
        posterior=None,
        return_sites="obs",
        device="cpu",
    ):
        super().__init__(model, posterior, return_sites, num_samples, device)

        self.warmup_steps = warmup_steps

    def _fit_approx(self, X_tensor, y_tensor=None):
        nuts_kernel = pyro.infer.NUTS(self.model)
        self.mcmc_ = pyro.infer.MCMC(nuts_kernel, num_samples=self.num_samples, warmup_steps=self.warmup_steps)
        self.mcmc_.run(X_tensor, y_tensor)
        return self

    def _predict_approx(self, X_tensor):
        predictive = pyro.infer.Predictive(
            model=self.model,
            posterior_samples=self.mcmc_.get_samples(),
            num_samples=self.num_samples,
            return_sites=(self.return_sites,),
            parallel=self.parallel_,
        )

        with torch.no_grad():
            samples = predictive(X_tensor)
        return samples
