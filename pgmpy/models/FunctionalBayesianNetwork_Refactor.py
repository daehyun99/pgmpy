from collections.abc import Hashable, Iterable

import networkx as nx
import pandas as pd

from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.models import DiscreteBayesianNetwork


class FunctionalBayesianNetwork(DiscreteBayesianNetwork):
    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
    ) -> None:
        super().__init__(
            ebunch=ebunch,
        )

    def add_cpds(self, *cpds: FunctionalCPD) -> None:
        for cpd in cpds:
            self.cpds.append(cpd)

    def fit(
        self,
        data: pd.DataFrame,
    ):

        sort_nodes = list(nx.topological_sort(self))

        for node in sort_nodes:
            cpd = next((c for c in self.cpds if c.variable == node), None)

            if cpd is None:
                continue

            parents = list(self.predecessors(node))
            required_columns = parents + [node]
            node_data = data[required_columns]

            cpd.fit(node_data, target=node, parents=parents)
