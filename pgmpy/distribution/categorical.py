from __future__ import annotations

import numpy as np

try:
    from skpro.distributions.base import BaseDistribution as _SkproBaseDistribution
except ImportError:  # pragma: no cover - optional dependency
    _SkproBaseDistribution = object


class Categorical(_SkproBaseDistribution):
    """Categorical distribution container compatible with CPD predictions."""

    def __init__(self, classes, probs):
        self.classes = list(classes)
        self.probs = np.asarray(probs, dtype=float)

        if self.probs.ndim == 1:
            self.probs = self.probs.reshape(1, -1)

        if self.probs.ndim != 2:
            raise ValueError("probs must be a 1D or 2D array-like object.")

        if self.probs.shape[1] != len(self.classes):
            raise ValueError("Number of probability columns must match number of classes.")

    def mean(self):
        return self.probs

    def mode(self):
        return np.array([self.classes[i] for i in np.argmax(self.probs, axis=1)])

    def sample(self, n_samples=1, random_state=None):
        """Sample class labels from each row-wise categorical distribution."""
        rng = np.random.default_rng(random_state)
        n_rows = self.probs.shape[0]
        samples = np.empty((n_rows, n_samples), dtype=object)
        for i in range(n_rows):
            samples[i, :] = rng.choice(self.classes, size=n_samples, p=self.probs[i])
        return samples

    def plot(self, row=0, ax=None):
        """Plot category probabilities for a selected row."""
        import matplotlib.pyplot as plt

        if not 0 <= row < self.probs.shape[0]:
            raise IndexError("row is out of bounds for probability matrix.")

        if ax is None:
            _, ax = plt.subplots()

        ax.bar(self.classes, self.probs[row])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title(f"Categorical distribution (row={row})")
        return ax
