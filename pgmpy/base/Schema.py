# from pgmpy.components import SCHEMA
class SCHEMA:
    def __init__(self):
        self._types = {}
        self._model = None

    def bind(self, model):
        self._model = model
        return self

    def add(self, variable, var_type):
        if var_type not in {"discrete", "continuous"}:
            raise ValueError(f"Unknown variable type: {var_type}")

        if self._model is not None:
            if variable not in self._model.graph.nodes:
                raise ValueError(f"{variable} is not in the graph.")

        self._types[variable] = var_type

    def copy(self):
        new = SCHEMA()
        new._types = self._types.copy()
        return new
