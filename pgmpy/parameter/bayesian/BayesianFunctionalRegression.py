from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy.distributions.converter.PyroToSkpro import PyroToSkpro
from pgmpy.parameter._base import BaseParameter

torch = _safe_import("torch")
pyro = _safe_import("pyro", pkg_name="pyro-ppl")


def to_tensor(arr, device, reshape=False):
    """
    helper function

    pd.DataFrame -> torch.tensor

    FYI: `pgmpy.utils.compat_fns` only support tensor <-> numpy

    """
    tensor = torch.as_tensor(
        arr.to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    if reshape:
        tensor = tensor.reshape(-1)

    return tensor


class BayesianFunctionalRegression(BaseParameter):
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
        estimator="svi",  # "mcmc"
        optim=None,
        loss=None,
        num_iterations=1500,
        posterior=None,
        device="cpu",
        log=True,
    ):
        # NOTE:
        #   Only support "dtype" == "torch.float32"

        super().__init__()
        _check_soft_dependencies(self.get_tag("python_dependencies"))

        if (estimator == "svi") and (guide is None):
            raise ValueError("You should set ...")

        self.model = model
        self.guide = guide
        self.estimator = estimator
        self.optim = optim
        self.loss = loss
        self.num_iterations = num_iterations
        self.posterior = posterior
        self.device = device
        self.log = log
        self._is_fitted = False

    def fit(self, X, y=None):
        X_tensor = to_tensor(X, self.device)
        y_tensor = to_tensor(y, self.device, reshape=True)

        if self.optim is None:
            optim = pyro.optim.Adam({"lr": 0.3})
        else:
            optim = self.optim
        if self.loss is None:
            loss = pyro.infer.Trace_ELBO()
        else:
            loss = self.loss

        pyro.clear_param_store()

        if self.estimator.lower() == "svi":
            svi = pyro.infer.SVI(
                model=self.model,
                guide=self.guide,
                optim=optim,
                loss=loss,
            )

            for j in range(self.num_iterations):
                loss = svi.step(X_tensor, y_tensor)

                if self.log and j % 100 == 0:
                    print(f"[iteration {j + 1:04d}] loss: {loss / X_tensor.shape[0]:.4f}")

        elif self.estimator.lower() == "mcmc":
            nuts_kernel = pyro.infer.NUTS(self.model)
            self.mcmc_ = pyro.infer.MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
            self.mcmc_.run(X_tensor, y_tensor)

        self._is_fitted = True
        self.columns_ = y.columns[0]
        if self.device == "cpu":
            self.parallel_ = False
        else:
            self.parallel_ = True
        return self

    def predict_proba(self, X):
        index = X.index
        X_tensor = to_tensor(X, self.device)

        if self.posterior is None:
            posterior = PyroToSkpro("normal", "obs", 2000)
        else:
            posterior = self.posterior

        if self.estimator.lower() == "svi":
            predictive = pyro.infer.Predictive(
                model=self.model,
                guide=self.guide,
                num_samples=posterior.num_samples,
                return_sites=(posterior.name,),
                parallel=self.parallel_,
            )

        elif self.estimator.lower() == "mcmc":
            predictive = pyro.infer.Predictive(
                model=self.model,
                posterior_samples=self.mcmc_.get_samples(),
                num_samples=posterior.num_samples,
                return_sites=(posterior.name,),
                parallel=self.parallel_,
            )

        with torch.no_grad():
            samples = predictive(X_tensor)

        SkproDistribution = posterior.convert(samples, index, [self.columns_], posterior.name)
        return SkproDistribution
