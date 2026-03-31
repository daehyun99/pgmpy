import numpy as np

from pgmpy.factors.continuous import LinearGaussianCPD


class LinearGaussianAdapter:
    """
    Adapter that fits data into a `LinearGaussianCPD`.
    """

    def __init__(self, variable, estimator=None, parents=None):
        self.variable = variable
        self.estimator = estimator
        self.parents = parents if parents is not None else []

    def fit(self, data):
        if self.estimator not in ("MLE", "OLS", None):
            raise ValueError(f"For linear tag, MLE/OLS is supported. Got {self.estimator}")

        target_data = data[self.variable].values

        if not self.parents:
            mean = np.mean(target_data)
            std = np.std(target_data)
            beta = [mean]
        else:
            evidence_data = data[self.parents].values
            X = np.c_[np.ones(evidence_data.shape[0]), evidence_data]

            beta, residuals, rank, s = np.linalg.lstsq(X, target_data, rcond=None)
            if len(residuals) > 0:
                variance = residuals[0] / len(target_data)
            else:
                predictions = X @ beta
                variance = np.mean((target_data - predictions) ** 2)
            std = np.sqrt(variance)

        self.fitted_cpd_ = LinearGaussianCPD(variable=self.variable, beta=beta, std=std, evidence=self.parents)
        return self

    def __repr__(self):
        if not hasattr(self, "fitted_cpd_"):
            return f"<LinearGaussianAdapter(variable='{self.variable}', status='unfitted') at {hex(id(self))}>"

        cpd = self.fitted_cpd_
        beta_str = f"{cpd.beta[0]:.3f}"
        for i, parent in enumerate(cpd.evidence):
            beta_str += f" + {cpd.beta[i + 1]:.3f}*{parent}"

        return (
            f"<LinearGaussianAdapter representing P({cpd.variable} | {', '.join(cpd.evidence)}) "
            f"~ N({beta_str}, std={cpd.std:.3f}) at {hex(id(self))}>"
        )
