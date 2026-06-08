# from pgmpy.components import PARAMETER
class PARAMETER:
    def __init__(self):
        self._cpds = {}
        self._model = None

    def bind(self, model):
        self._model = model
        return self

    def add(self, variable, cpd):
        if self._model is not None:
            if variable not in self._model.graph.nodes:
                raise ValueError(f"{variable} is not in the graph.")

        self._cpds[variable] = cpd

    def copy(self):
        new = PARAMETER()
        new._cpds = self._cpds.copy()
        return new
