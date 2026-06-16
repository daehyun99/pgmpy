from pgmpy.parameter._base import BaseParameter

class TabularCPD(BaseParameter):
    """TabularCPD"""

    _tags = {
        "variable_type": "discrete",
        "produces_factor": True,
        "is_linear_gaussian": False,
        "missing": False,
        "supports_fit_joint": False,
        "python_dependencies": ("skpro"),
    }

    def __init__(
        self,
        variable_card,
        evidence_card=None,
        state_names=None,
        prior_type=None,
        equivalent_sample_size=10,
        pseudo_counts=None
    ):
        ...

    def _fit(self, X, y, sample_weight=None):
        self.values_ = ...
        self.state_names_ = ...
        self.columns_ = ...
        self.is_fitted_=True

        return self
    
    def _predict_proba(self, X):
        return ... # CategoricalDistribution # (len(X), variable_card)

    def get_values(
        self,
        values,
        evidence_card=None,
        state_names=None,
        parent_order=None,
    ):
        cls.is_fitted_ = True
        return TabularCPD(...)
