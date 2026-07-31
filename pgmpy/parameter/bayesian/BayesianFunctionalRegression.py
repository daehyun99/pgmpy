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
    if reshape:
        arr_tensor = torch.as_tensor(
            arr.to_numpy().reshape(-1),
            dtype=torch.float32,
            device=device,
        )
    else:
        arr_tensor = torch.tensor(
            arr.to_numpy(),
            dtype=torch.float32,
            device=device,
        )
    return arr_tensor


class BayesianFunctionalRegression(BaseParameter):
    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": False,
        "missing": False,
        "supports_fit_joint": False,
        "can_be_root": False,
        "python_dependencies": ["pyro-ppl"],
    }

    def __init__(
        self,
        model,
        guide,
        optimizer=pyro.optim.Adam({"lr": 0.3}),
        loss=pyro.infer.Trace_ELBO(),
        num_iterations=1500,
        posterior_name="obs",
        posterior_type="normal",
        posterior_samples=2000,
        device="cpu",
        log=True,
    ):
        # NOTE:
        #   Only support "pyro.infer.SVI"
        #   Only support "posterior_type" == "normal"
        #   Only support "optimizer" == "pyro.optim.Adam"
        #   Only support "loss" == "pyro.infer.Trace_ELBO"
        #   Only support "dtype" == "torch.float32"

        super().__init__()
        _check_soft_dependencies(self.get_tag("python_dependencies"))

        if (not model) and (not guide):
            raise ValueError("You should set ...")

        self.model = model
        self.guide = guide
        self.optimizer = optimizer
        self.loss = loss
        self.num_iterations = num_iterations
        self.posterior_name = posterior_name
        self.posterior_type = posterior_type
        self.posterior_samples = posterior_samples
        self.device = device
        self.log = log

        self.converter_ = PyroToSkpro(self.posterior_type, self.device)
        self._is_fitted = False

    def fit(self, X, y):
        X_tensor = to_tensor(X, self.device)
        y_tensor = to_tensor(y, self.device, reshape=True)

        pyro.clear_param_store()

        svi = pyro.infer.SVI(
            model=self.model,
            guide=self.guide,
            optim=self.optimizer,
            loss=self.loss,
        )

        for j in range(self.num_iterations):
            loss = svi.step(X_tensor, y_tensor)

            if self.log and j % 100 == 0:
                print(f"[iteration {j + 1:04d}] loss: {loss / X_tensor.shape[0]:.4f}")

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

        predictive = pyro.infer.Predictive(
            model=self.model,
            guide=self.guide,
            num_samples=self.posterior_samples,
            return_sites=(self.posterior_name,),
            parallel=self.parallel_,
        )

        with torch.no_grad():
            samples = predictive(X_tensor)

        SkproDistribution = self.converter_.convert(samples, index, [self.columns_], self.posterior_name)
        return SkproDistribution
