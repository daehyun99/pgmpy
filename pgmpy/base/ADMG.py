from collections.abc import Hashable, Iterable

from pgmpy.base._base import _CoreGraph


class ADMG(_CoreGraph):
    """
    A class representing an Acyclic Directed Mixed Graph (ADMG).

    An ADMG is a directed graph that allows for both directed and bidirected edges.

    Parameters
    ----------
    edge_list : iterable of tuples, optional
        A list or iterable of edges to add at initialization.

    latents : set of nodes, (default=set())
        A set of latent variables in the graph. These are not observed
        variables but are used to represent unobserved confounding or
        other latent structures.

    exposures : set, (default=set())
        Set of exposure variables in the graph. These are the variables
        that represent the treatment or intervention being studied in a
        causal analysis. Default is an empty set.

    outcomes : set, (default=set())
        Set of outcome variables in the graph. These are the variables
        that represent the response or dependent variables being studied
        in a causal analysis. Default is an empty set.

    roles : dict, optional (default=None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).

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

    def _validate_graph_specific_edges(self, edge_list):
        return super()._validate_graph_specific_edges(edge_list)

    def is_multigraph(self):
        return False

    def is_acyclic(self):
        return True
