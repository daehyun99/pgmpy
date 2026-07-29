import matplotlib.pyplot as plt
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
from torch import nn

from pgmpy.parameter._base import BaseParameter

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
        guide,
        converter,
        num_iterations=1500,
        lr=0.03,
        posterior_samples=2000,
    ):
        super().__init__()

        self.num_iterations = num_iterations
        self.lr = lr
        self.posterior_samples = posterior_samples

        self.model = model if model else None
        self.guide = guide if guide else None
        self.converter = converter if converter else None

        self._is_fitted = False

    def fit(self, X, y):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(
                X.to_numpy(),
                dtype=torch.float32
            )

            y_tensor = torch.tensor(
                y.to_numpy().reshape(-1),
                dtype=torch.float32
            )
        else:
            X_tensor, y_tensor = X, y

        pyro.clear_param_store()

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

        pred_dist = self.converter(y_samples, num_samples, X_array, index=index, columns=pd.Index(["y"]))
        return pred_dist
