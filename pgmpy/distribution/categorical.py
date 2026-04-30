import numpy as np


class Categorical:
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
