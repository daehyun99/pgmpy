from collections.abc import Hashable, Iterable

from pgmpy.base._base import _CoreGraph


class ADMG(_CoreGraph):
    """
    A class representing an Acyclic Directed Mixed Graph (ADMG).

    An ADMG is a directed graph that allows for both directed and bidirected edges.
    This class extends the `networkx.MultiDiGraph` and provides additional functionality
    for operations involving directed and bidirected edges.

    Parameters
    ----------
    directed_edge_list : list of tuple, optional
        List of directed edges to initialize the graph, where each tuple is (u, v).
    bidirected_edge_list : list of tuple, optional
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
        edge_list: Iterable[tuple[Hashable, Hashable, Hashable]] = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        latents: set[Hashable] | None = None,
        roles=None,
    ):
        super().__init__(
            edge_list=edge_list,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

    def get_district(self, nodes: Hashable) -> set:
        """
        Return district of a node: maximal set connected via bidirected edges.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        components: set
            Nodes in the same bidirected-connected component.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG()
        >>> admg.add_edges_from(
        ...     [
        ...         ("A", "B", "->"),
        ...         ("B", "C", "->"),
        ...         ("D", "B", "->"),
        ...         ("A", "D", "<>"),
        ...         ("B", "E", "<>"),
        ...     ]
        ... )
        >>> sorted(admg.get_district("A"))
        ['A', 'D']
        >>> sorted(admg.get_district("B"))
        ['B', 'E']

        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        components = set()

        for node in nodes_set:
            component = self.get_reachable_nodes(node, edge_type="<>")
            components.update(component)

        return components

    # def to_dag(self):
    #     """

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     Graphs

    #     See Also
    #     --------
    #     `DAG`, `ADMG`

    #     Notes
    #     -----

    #     Examples
    #     --------

    #     References
    #     ----------

    #     """
    #     # TODO(@daehyun99): [#2385] Implement method when Refactor DAG
    #     # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
    #     # TODO(@daehyun99): [#2385] Apply type hint(input, output)
    #     raise NotImplementedError("`to_dag` is not supported now")

    # def is_valid_admg(self):
    #     """
    #     checking is admg
    #     - the graph does not contain any directed cycles

    #     Parameters
    #     ----------
    #     None

    #     Returns
    #     -------
    #     bool

    #     See Also
    #     --------
    #     `MAG`, `PAG`

    #     Notes
    #     -----

    #     Examples
    #     --------

    #     References
    #     ----------
    #     [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
    #     Journal of Machine Learning Research 9 (2008): 1437-1474.
    #     """
    #     # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
    #     # TODO(@daehyun99): [#2385] Apply type hint(input, output)
    #     # # TODO(@daehyun99): [#2385] Implement code logic and test code When Refactor DAG
    #     # if self.has_directed_cycle():
    #     # return False
    #     # TODO(@daehyun99): [#2385] Checking edge type(direct, bidirect)
    #     # return True
    #     raise NotImplementedError("`is_valid_admg` is not supported now")

    def _validate_graph_specific_edges(
        self,
        edge_list: (
            Iterable[tuple[Hashable, Hashable, Hashable]] | Iterable[tuple[Hashable, Hashable, Hashable, Hashable]]
        ),
    ):
        for edge in edge_list:
            if len(edge) == 3:
                u, v, edge_type = edge
            elif len(edge) == 4:
                u, v, _, edge_type = edge
        if edge_type == "->":
            if self.has_node(u) and self.has_node(v) and self.has_direct_path(v, u):
                raise ValueError("Cycles are not allowed in a ADMG.")
        elif edge_type == "<-":
            if self.has_node(u) and self.has_node(v) and self.has_direct_path(u, v):
                raise ValueError("Cycles are not allowed in a ADMG.")

    def is_multigraph(self):
        return True
