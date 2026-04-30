from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from typing import Any

from pgmpy.base import DAG


class BayesianNetwork(DAG):
    """Graph container for Bayesian network structure and CPD metadata.

    This class intentionally focuses on model-definition responsibilities only
    (nodes, edges, roles, latent/exposure/outcome annotations, and CPD storage).
    Learning, inference, prediction, and simulation should be handled by
    estimator/inference components that consume this model.
    """

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        latents: set[Hashable] | None = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        roles: dict[str, Iterable] | None = None,
    ) -> None:
        super().__init__(
            ebunch=ebunch,
            latents=latents,
            exposures=exposures,
            outcomes=outcomes,
            roles=roles,
        )
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_cpds(self, *cpds: Any) -> None:
        """Add CPDs to the model.

        CPD objects are expected to provide at least a `variable` attribute.
        If present, `variables` and `cardinality` are used to update
        `self.cardinalities`.
        """
        for cpd in cpds:
            if not hasattr(cpd, "variable"):
                raise ValueError("Only CPD-like objects with a `variable` attribute can be added.")

            if cpd.variable not in self.nodes():
                raise ValueError(f"CPD defined on variable not in the model: {cpd.variable}")

            for index, existing_cpd in enumerate(self.cpds):
                if existing_cpd.variable == cpd.variable:
                    self.cpds[index] = cpd
                    break
            else:
                self.cpds.append(cpd)

            variables = getattr(cpd, "variables", [])
            cardinality = getattr(cpd, "cardinality", [])
            if variables and cardinality:
                for variable, card in zip(variables, cardinality):
                    self.cardinalities[variable] = card

    def get_cpds(self, node: Hashable | None = None) -> Any:
        """Return CPD(s) in the model or CPD associated with a node."""
        if node is None:
            return self.cpds

        if node not in self.nodes():
            raise ValueError("Node not present in the graph")

        for cpd in self.cpds:
            if cpd.variable == node:
                return cpd
        return None

    def check_model(self) -> bool:
        """Validate that each node has a CPD and CPDs are structurally consistent."""
        for node in self.nodes():
            cpd = self.get_cpds(node=node)
            if cpd is None:
                raise ValueError(f"No CPD associated with {node}")

            evidence = set(getattr(cpd, "get_evidence", lambda: [])())
            parents = set(self.get_parents(node))
            if evidence != parents:
                raise ValueError(
                    f"CPD associated with {node} doesn't have proper parents associated with it."
                )

            if hasattr(cpd, "is_valid_cpd") and not cpd.is_valid_cpd():
                raise ValueError(f"Sum or integral of conditional probabilities for node {node} is not equal to 1.")

        return True
