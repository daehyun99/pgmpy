import matplotlib.pyplot as plt
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
from skpro.distributions.normal import Normal as SkproNormal
from torch import nn

from pgmpy.parameter._base import BaseParameter

pyro.set_rng_seed(1)
plt.style.use("default")


class _BayesianLinearRegression(PyroModule):
    def __init__(self, in_features, mu=0.0, sigma=1.0):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, 1)
        self.linear.weight = PyroSample(dist.Normal(mu, sigma).expand([1, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(mu, sigma).expand([1]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


class BayesianLinearRegression(BaseParameter):
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
        in_features,
        num_iterations=1500,
        lr=0.03,
        posterior_samples=2000,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_iterations = num_iterations
        self.lr = lr
        self.posterior_samples = posterior_samples

        self.model = None
        self.guide = None
        self._is_fitted = False

    def fit(self, X, y):
        X_tensor = torch.tensor(
            X.to_numpy(),
            dtype=torch.float32
        )

        y_tensor = torch.tensor(
            y.to_numpy(),
            dtype=torch.float32
        )
        pyro.clear_param_store()
        self.model = _BayesianLinearRegression(self.in_features)
        self.guide = AutoDiagonalNormal(self.model)

        optimizer = pyro.optim.Adam({"lr": self.lr})
        svi = SVI(
            self.model,
            self.guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        for j in range(self.num_iterations):
            loss = svi.step(X_tensor, y_tensor)

            if j % 100 == 0:
                print(f"[iteration {j + 1:04d}] loss: {loss / X_tensor.shape[0]:.4f}")

        self.guide.requires_grad_(False)
        self._is_fitted = True
        self.evidences_=list(X.columns)
        return self

    def predict_proba(self, X):
        num_samples = self.posterior_samples
        X_tensor = torch.tensor(
            X.to_numpy(),
            dtype=torch.float32
        )
        X_array = X_tensor.numpy()
        index = X.index

        guide_param = next(self.guide.parameters())
        X_tensor = torch.as_tensor(
            X_array,
            dtype=guide_param.dtype,
            device=guide_param.device,
        )

        predictive = Predictive(
            model=self.model,
            guide=self.guide,
            num_samples=num_samples,
            return_sites=("obs",),
            parallel=False,
        )

        with torch.no_grad():
            samples = predictive(X_tensor)
            y_samples = samples["obs"]

        y_samples = y_samples.reshape(num_samples, X_array.shape[0])

        pred_mean = y_samples.mean(dim=0)
        pred_sigma = y_samples.std(dim=0, unbiased=False)

        eps = torch.finfo(pred_sigma.dtype).eps
        pred_sigma = pred_sigma.clamp_min(eps)

        pred_mean = pred_mean.detach().cpu().numpy()
        pred_sigma = pred_sigma.detach().cpu().numpy()

        return SkproNormal(
            mu=pred_mean.reshape(-1, 1),
            sigma=pred_sigma.reshape(-1, 1),
            index=index,
            columns=pd.Index(["y"]),
        )
