import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pyro = pytest.importorskip("pyro")
dist = pytest.importorskip("pyro.distributions")


from pgmpy.parameter.bayesian._base import BasePyroRegression


@pytest.fixture
def data():
    pyro.set_rng_seed(42)
    rng = np.random.default_rng(42)

    beta = 0.8
    intercept = 0.3
    noise_scale = 0.7
    n_samples = 1000

    x1 = rng.normal(0.2, 0.9, size=n_samples)
    x2 = rng.normal((x1 * 0.5) + 0.5, 0.6)
    x3 = rng.normal((x2 * beta) + intercept, noise_scale)

    data = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )

    X = data[["x2"]]
    y = data[["x3"]]
    return X, y


@pytest.fixture
def model():
    def model(X, y=None):
        device = X.device

        beta = pyro.sample(
            "beta",
            dist.Normal(
                torch.tensor(0.0, device=device),
                torch.tensor(2.0, device=device),
            ),
        )

        intercept = pyro.sample(
            "intercept",
            dist.Normal(
                torch.tensor(0.0, device=device),
                torch.tensor(2.0, device=device),
            ),
        )

        sigma = pyro.sample(
            "sigma",
            dist.LogNormal(
                torch.tensor(0.0, device=device),
                torch.tensor(1.0, device=device),
            ),
        )

        mean = intercept + beta * X[:, 0]

        with pyro.plate("data", X.shape[0]):
            pyro.sample(
                "obs",
                dist.Normal(mean, sigma),
                obs=y,
            )

    return model


class TestBasePyroRegression:
    def test_base_parameter_default(self, model):
        parameter = BasePyroRegression(model)

        assert parameter.__class__.__name__ == "BasePyroRegression"
        assert parameter.get_class_tag("variable_type") == "continuous"
        assert parameter.get_class_tag("produces_factor") is False
        assert parameter.get_class_tag("missing") is False
        assert parameter.get_class_tag("is_linear_gaussian") is False
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("can_be_root") is False
        assert parameter.get_class_tag("python_dependencies") == ("pyro-ppl")

    def test_fails(self, data, model):
        X, y = data
        with pytest.raises(TypeError):
            BasePyroRegression()

        with pytest.raises(TypeError):
            parameter = BasePyroRegression(model)
            parameter.fit()
