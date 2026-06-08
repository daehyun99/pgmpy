class CounterfactualInference:
    def __init__(self):
        # super().__init__()
        ...

    def bind(self, model):
        self._model = model
        return self

    def query(self, evidence: dict, do: dict, seed: int): ...
