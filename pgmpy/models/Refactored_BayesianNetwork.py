from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pgmpy.base import DAG


@dataclass
class NodeObject:
    node: Hashable
    parents: set[Hashable] = field(default_factory=set)
    children: set[Hashable] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    local_model: Any = None
    estimator: Any = None
    data: Any = None


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
        self.nodedict: dict[Hashable, NodeObject] = {}
        for node in self.nodes():
            self.nodedict[node] = NodeObject(node=node)

    def add_cpds(self, *cpds: Any) -> None:
        """Add CPDs to the model.

        CPD objects are expected to provide at least a `variable` attribute.
        If present, `variables` and `cardinality` are used to update
        `self.cardinalities`.
        """
        for cpd in cpds:
            self.cpds.append(cpd)
            if cpd.variable in self.nodedict:
                self.nodedict[cpd.variable].local_model = cpd

    def add_node(self, node_for_adding: Hashable, **attr: Any) -> None:
        super().add_node(node_for_adding, **attr)
        self.nodedict.setdefault(node_for_adding, NodeObject(node=node_for_adding))

    def add_edge(self, u: Hashable, v: Hashable, **attr: Any) -> None:
        super().add_edge(u, v, **attr)
        self.nodedict.setdefault(u, NodeObject(node=u))
        self.nodedict.setdefault(v, NodeObject(node=v))

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

    def get_node(self, node: Hashable) -> NodeObject:
        if node not in self.nodes():
            raise ValueError("Node not present in the graph")

        roles = {role for role in self[node].keys() if self[node][role]}
        node_obj = self.nodedict.get(node, NodeObject(node=node))
        node_obj.parents = set(self.predecessors(node))
        node_obj.children = set(self.successors(node))
        node_obj.roles = roles
        node_obj.local_model = self.get_cpds(node=node)
        self.nodedict[node] = node_obj
        return node_obj

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
