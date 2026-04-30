import numpy as np
import pandas as pd
from skbase.base import BaseEstimator

from pgmpy.distribution import Categorical
from pgmpy.factors.discrete import TabularCPD
from pgmpy.parameter_estimator import DiscreteMLE


def test_predict_proba_returns_categorical_distribution():
    cpd = TabularCPD(
        variable="grade",
        variable_card=2,
        values=[[0.7, 0.2], [0.3, 0.8]],
        evidence=["diff"],
        evidence_card=[2],
        state_names={"grade": ["A", "B"], "diff": ["easy", "hard"]},
    )

    X = pd.DataFrame({"diff": ["easy", "hard", "hard"]})
    pred_dist = cpd.predict_proba(X)

    assert isinstance(pred_dist, Categorical)
    np.testing.assert_allclose(pred_dist.mean(), np.array([[0.7, 0.3], [0.2, 0.8], [0.2, 0.8]]))


def test_predict_returns_mode_states():
    cpd = TabularCPD(
        variable="grade",
        variable_card=2,
        values=[[0.7, 0.2], [0.3, 0.8]],
        evidence=["diff"],
        evidence_card=[2],
        state_names={"grade": ["A", "B"], "diff": ["easy", "hard"]},
    )

    X = pd.DataFrame({"diff": ["easy", "hard"]})
    pred = cpd.predict(X)

    assert list(pred) == ["A", "B"]


def test_fit_updates_cpd_from_data():
    cpd = TabularCPD(
        variable="grade",
        variable_card=2,
        values=[[0.5, 0.5], [0.5, 0.5]],
        evidence=["diff"],
        evidence_card=[2],
        state_names={"grade": ["A", "B"], "diff": ["easy", "hard"]},
    )

    data = pd.DataFrame(
        {
            "diff": ["easy", "easy", "hard", "hard", "hard"],
            "grade": ["A", "A", "A", "B", "B"],
        }
    )

    cpd.fit(data)

    expected = np.array([[1.0, 1 / 3], [0.0, 2 / 3]])
    np.testing.assert_allclose(cpd.get_values(), expected)


def test_fit_delegates_to_discrete_mle_estimator(monkeypatch):
    cpd = TabularCPD(
        variable="grade",
        variable_card=2,
        values=[[0.5, 0.5], [0.5, 0.5]],
        evidence=["diff"],
        evidence_card=[2],
        state_names={"grade": ["A", "B"], "diff": ["easy", "hard"]},
    )
    data = pd.DataFrame({"diff": ["easy", "hard"], "grade": ["A", "B"]})
    called = {"value": False}

    original_estimate_cpd = DiscreteMLE._estimate_cpd

    def wrapped(*args, **kwargs):
        called["value"] = True
        return original_estimate_cpd(*args, **kwargs)

    monkeypatch.setattr(DiscreteMLE, "_estimate_cpd", wrapped)
    cpd.fit(data)
    assert called["value"]


def test_tabular_cpd_is_base_estimator():
    cpd = TabularCPD(
        variable="grade",
        variable_card=2,
        values=[[0.7], [0.3]],
        state_names={"grade": ["A", "B"]},
    )

    assert isinstance(cpd, BaseEstimator)

def test_categorical_sample_shape_and_values():
    dist = Categorical(classes=["A", "B"], probs=[[0.7, 0.3], [0.2, 0.8]])
    samples = dist.sample(n_samples=5, random_state=7)

    assert samples.shape == (2, 5)
    assert set(np.unique(samples)).issubset({"A", "B"})


def test_categorical_plot_returns_axes():
    import matplotlib

    matplotlib.use("Agg")

    dist = Categorical(classes=["A", "B"], probs=[0.7, 0.3])
    ax = dist.plot()

    assert ax.get_ylabel() == "Probability"
