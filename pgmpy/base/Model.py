from pgmpy.base.Schema import SCHEMA


class Model:
    _COMPONENT_NAMES = (
        "schema",
        "parameter",
        "inference",
        "intervene",
        "counterfactual",
    )

    def __init__(
        self,
        graph=None,
        schema=None,
        parameter=None,
        inference=None,
        intervene=None,
        counterfactual=None,
        copy=True,
    ):
        self.graph = self._copy_if_needed(graph, copy)
        self._bind_if_possible(self.graph)

        components = {
            "schema": schema,
            "parameter": parameter,
            "inference": inference,
            "intervene": intervene,
            "counterfactual": counterfactual,
        }

        for name, component in components.items():
            component = self._copy_if_needed(component, copy)
            component = self._bind_if_possible(component)
            setattr(self, f"_{name}", component)

    def _copy_if_needed(self, obj, copy):
        if copy and obj is not None:
            return obj.copy()
        return obj

    def _bind_if_possible(self, obj):
        if obj is not None:
            return obj.bind(self)
        return obj

    def _get_component(self, name, default_factory=None):
        attr_name = f"_{name}"
        component = getattr(self, attr_name)

        if component is None and default_factory is not None:
            component = default_factory()
            component = self._bind_if_possible(component)
            setattr(self, attr_name, component)

        return component

    def _set_component(self, name, value):
        setattr(self, f"_{name}", self._bind_if_possible(value))

    @property
    def schema(self):
        return self._get_component("schema", SCHEMA)

    @schema.setter
    def schema(self, value):
        self._set_component("schema", value)

    @property
    def parameter(self):
        return self._get_component("parameter")

    @parameter.setter
    def parameter(self, value):
        self._set_component("parameter", value)

    @property
    def inference(self):
        return self._get_component("inference")

    @inference.setter
    def inference(self, value):
        self._set_component("inference", value)

    @property
    def intervene(self):
        return self._get_component("intervene")

    @intervene.setter
    def intervene(self, value):
        self._set_component("intervene", value)

    @property
    def counterfactual(self):
        return self._get_component("counterfactual")

    @counterfactual.setter
    def counterfactual(self, value):
        self._set_component("counterfactual", value)
