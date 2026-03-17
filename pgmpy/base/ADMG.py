from typing import Hashable, Iterable, Optional

from pgmpy.base._base import _CoreGraph


class ADMG(_CoreGraph):
    """
    A class representing an Acyclic Directed Mixed Graph (ADMG).

    An ADMG is a directed graph that allows for both directed and bidirected edges.
    This class extends the `networkx.MultiDiGraph` and provides additional functionality
    for operations involving directed and bidirected edges.

    Parameters
    ----------
    directed_ebunch : list of tuple, optional
        List of directed edges to initialize the graph, where each tuple is (u, v).
    bidirected_ebunch : list of tuple, optional
        List of bidirected edges to initialize the graph, where each tuple is (u, v).
    latents : set of str, optional
        Set of latent variables in the graph. These are not directly represented as nodes
        but are used to indicate the presence of bidirected edges.
    roles : dict, optional (default: None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).
        If provided, this will automatically assign roles to the nodes in the graph.
        Passing a key-value pair via ``roles`` is equivalent to calling
        ``with_role(role, variables)`` for each key-value pair in the dictionary.

    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "<>"])

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable, Hashable]] = None,
        exposures: Optional[set[Hashable]] = None,
        outcomes: Optional[set[Hashable]] = None,
        latents: Optional[set[Hashable]] = None,
        roles=None,
    ):
        super().__init__(
            ebunch=ebunch,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

    def add_edge(self, u, v, edge_type="->", key=None, **kwargs):
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        # NOTE: No additional comments are needed, as the comments in _CoreGraph are utilized.

        if edge_type == "->":
            if self.has_node(u) and self.has_node(v) and self.has_direct_path(v, u):
                raise ValueError("Cycles are not allowed in a ADMG.")
        elif edge_type == "<-":
            if self.has_node(u) and self.has_node(v) and self.has_direct_path(u, v):
                raise ValueError("Cycles are not allowed in a ADMG.")
        super().add_edge(u, v, edge_type, key, **kwargs)

    def add_edges_from(self, ebunch, **kwargs):
        # NOTE: No additional comments are needed, as the comments in _CoreGraph are utilized.
        self._validate_edges(ebunch=ebunch)
        for edge in ebunch:
            if len(edge) == 3:
                u, v, edge_type = edge
                self.add_edge(u, v, edge_type=edge_type, **kwargs)
            elif len(edge) == 4:
                u, v, key, edge_type = edge
                self.add_edge(u, v, edge_type=edge_type, key=key, **kwargs)

    def get_district(self, nodes):
        """
        Return district of a node: maximal set connected via bidirected edges.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Nodes in the same bidirected-connected component.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y")], bidirected_ebunch=[("X", "Z")])
        >>> sorted(admg.get_district("X"))
        ['X', 'Z']
        >>> admg.get_district("Y")
        {'Y'}
        """
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        components = set()

        for node in nodes_set:
            component = self.get_reachable_nodes(node, edge_type="<>")
            components.update(component)

        return components

    def to_dag(self):
        """

        Parameters
        ----------

        Returns
        -------
        Graphs

        See Also
        --------
        `DAG`, `ADMG`

        Notes
        -----

        Examples
        --------

        References
        ----------

        """
        # TODO(@daehyun99): [#2385] Implement method when Refactor DAG
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        raise NotImplementedError("`to_dag` is not supported now")
