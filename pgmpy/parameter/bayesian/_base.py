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


class BasePyroRegression(BaseParameter):
    """Base class for parameter classes using `pyro` in pgmpy."""

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
        posterior=None,
        return_sites="obs",
        num_samples=1000,
        device="cpu",
    ):
        super().__init__()
        _check_soft_dependencies(self.get_tag("python_dependencies"))

        self.model = model
        self.posterior = posterior
        self.return_sites = return_sites
        self.num_samples = num_samples
        self.device = device

    def _fit(self, X, y=None, sample_weight=None):
        X_tensor = to_tensor(X, self.device)
        y_tensor = to_tensor(y, self.device, reshape=True)

        pyro.clear_param_store()
        self._fit_approx(X_tensor, y_tensor)

        self.columns_ = y.columns[0]
        if self.device == "cpu":
            self.parallel_ = False
        else:
            self.parallel_ = True
        return self

    def _fit_approx(X_tensor, y_tensor=None):
        raise NotImplementedError

    def _predict_proba(self, X):
        index = X.index
        X_tensor = to_tensor(X, self.device)

        if self.posterior is None:
            posterior = PyroToSkpro("normal")
        else:
            posterior = self.posterior

        samples = self._predict_approx(X_tensor)

        SkproDistribution = posterior.convert(samples, index, [self.columns_], self.return_sites)
        return SkproDistribution

    def _predict_approx(X_tensor):
        raise NotImplementedError
