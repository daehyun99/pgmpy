from skbase.utils.dependencies import _safe_import

from pgmpy.parameter.bayesian._base import BasePyroRegression

torch = _safe_import("torch")
pyro = _safe_import("pyro", pkg_name="pyro-ppl")


class BayesianMCMCRegression(BasePyroRegression):
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
