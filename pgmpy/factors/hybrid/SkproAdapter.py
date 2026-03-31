class SkproAdapter:
    """
    Minimal adapter that wraps external skpro-like estimators.
    """

    def __init__(self, variable, model, parents=None):
        self.variable = variable
        self.model = model
        self.parents = parents if parents is not None else []

    def fit(self, data):
        X = data[self.parents] if self.parents else data.iloc[:, 0:0]
        y = data[self.variable]
        self.model.fit(X, y)
        return self

    def __repr__(self):
        model_name = type(self.model).__name__
        return (
            f"<SkproAdapter(variable='{self.variable}', "
            f"model='{model_name}', parents={self.parents}) at {hex(id(self))}>"
        )
