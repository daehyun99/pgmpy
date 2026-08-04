from skbase.utils.dependencies import _safe_import

from pgmpy.parameter.bayesian._base import BasePyroRegression

torch = _safe_import("torch")
pyro = _safe_import("pyro", pkg_name="pyro-ppl")


class BayesianSVIRegression(BasePyroRegression):
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
        guide=None,
        optim=None,
        loss=None,
        num_iterations=1500,
        num_samples=1000,
        posterior=None,
        return_sites="obs",
        device="cpu",
        log=True,
    ):
        super().__init__(model, posterior, return_sites, num_samples, device)

        self.guide = guide
        self.optim = optim
        self.loss = loss
        self.num_iterations = num_iterations
        self.log = log

    def _fit_approx(self, X_tensor, y_tensor=None):
        if self.guide is None:
            guide = pyro.infer.autoguide.AutoNormal(self.model)
        else:
            guide = self.guide

        if self.optim is None:
            optim = pyro.optim.Adam({"lr": 0.3})
        else:
            optim = self.optim
        if self.loss is None:
            loss = pyro.infer.Trace_ELBO()
        else:
            loss = self.loss

        svi = pyro.infer.SVI(
            model=self.model,
            guide=guide,
            optim=optim,
            loss=loss,
        )

        for j in range(self.num_iterations):
            loss = svi.step(X_tensor, y_tensor)

            if self.log and j % 100 == 0:
                print(f"[iteration {j + 1:04d}] loss: {loss / X_tensor.shape[0]:.4f}")

        return self

    def _predict_approx(self, X_tensor):

        predictive = pyro.infer.Predictive(
            model=self.model,
            guide=self.guide,
            num_samples=self.num_samples,
            return_sites=(self.return_sites,),
            parallel=self.parallel_,
        )

        with torch.no_grad():
            samples = predictive(X_tensor)
        return samples
