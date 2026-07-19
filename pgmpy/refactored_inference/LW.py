from ._base import BaseInference


class LikelihoodWeighting(BaseInference):
    _tags = {...}

    def __init__(self):
        super().__init__()

    def query(self, model, variables, evidence=None, do=None): ...
