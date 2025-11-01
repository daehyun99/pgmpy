"""
Base class for pgmpy graph objects.

- Class comment: Comment for users.
- `__init__` method comment: Comment for developers.

"""

from typing import Hashable, Iterable, Optional, Union

import networkx as nx

from pgmpy.base._mixin_roles import _GraphRolesMixin


class _CoreGraph(nx.MultiDiGraph, _GraphRolesMixin):
    """
    Base graph class.

    Parameters
    ----------
    ebunch : input graph (optional, default: `None`)
            Each edge given in the container will be added to the
            graph. The edges can be:

                - 3-tuples (u, v, type)
                - 4-tuples (u, v, type, key) for an edge with data and key

    latents : set of nodes (default: empty `set()`)
    exposures : set of nodes (default: empty `set()`)
    outcomes : set of nodes (default: empty `set()`)
    roles : dict, optional (default: `None`)

    Examples
    --------
    Create an empty `_CoreGraph` with no nodes and no edges.

    >>> from pgmpy.base import _CoreGraph
    >>> G = _CoreGraph()

    Edges and vertices can be passed to the constructor as an edge list.

    >>> edges = [("A", "B", "->"), ("B", "C", "->")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> G.edges
    MultiEdgeView([('A', 'B', 0), ('B', 'C', 0)])
    >>> G.edges(data=True)
    MultiEdgeDataView([('A', 'B', {'type': '->'}), ('B', 'C', {'type': '->'})])
    >>> G.SUPPORTED_EDGE_TYPES  # check the available edge types
    ['--', '-o', 'o-', '->', '<-', 'o>', '<o', '<>', 'oo']

    **Nodes:**

    Add one node,

    >>> from pgmpy.base import _CoreGraph
    >>> G = _CoreGraph()
    >>> G.add_node("A")
    >>> G.nodes
    NodeView(('A',))

    Add a list of nodes,

    >>> G.add_nodes_from(["B", "C"])
    >>> G.nodes
    NodeView(('A', 'B', 'C'))

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> from pgmpy.base import _CoreGraph
    >>> G = _CoreGraph()
    >>> G.add_edge("A", "B", "->")
    >>> G.edges  # You can check the key value of the two nodes and the edge connecting the two nodes.
    MultiEdgeView([('A', 'B', 0)])
    >>> G.edges(
    ...     data=True
    ... )  # You can check the type value of the two nodes and the edge connecting the two nodes.
    MultiEdgeDataView([('A', 'B', {'type': '->'})])

    Add a list of edges,

    >>> edges = [("A", "B", "->"), ("B", "C", "->")]
    >>> G = _CoreGraph()
    >>> G.add_edges_from(ebunch=edges)
    >>> G.edges(data=True)
    MultiEdgeDataView([('A', 'B', {'type': '->'}), ('B', 'C', {'type': '->'})])

    Remove one edge,

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> G.remove_edge("A", "B", "->")
    >>> G.edges(data=True)
    MultiEdgeDataView([('B', 'C', {'type': '->'}), ('C', 'D', {'type': '--'})])

    Remove a list of edges,

    >>> remove_edges = [("B", "C", "->"), ("C", "D", "--")]
    >>> G.remove_edges_from(ebunch=remove_edges)
    >>> G.edges(data=True)
    MultiEdgeDataView([])

    **Exposures, Outcomes, and Latents:**
        We provide a way to easily add frequently used node roles, such as `exposure`, `outcomes`, and `latents`.

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("D", "C", "-o")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> G.exposures = "A"  # Add node 'A' with role 'exposure'
    >>> G.outcomes = "C"  # Add node 'C' with role 'outcomes'
    >>> G.latents = "D"  # Add node 'D' with role 'latents'
    >>> G.exposures  # Checks for the 'exposure' role node.
    {'A'}
    >>> G.outcomes  # Checks for the 'outcomes' role node.
    {'C'}
    >>> G.latents  # Checks for the 'latents' role node.
    {'D'}

    **Roles:**
        In addition to 'exposure', 'outcomes', and 'latents', you can add custom roles.

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("D", "C", "-o")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> G.with_role("Custom_role", "A", inplace=True)
    >>> G.with_role("latents", "D", inplace=True)
    >>> G.get_role_dict()
    {'latents': ['D'], 'Custom_role': ['A']}
    >>> G.without_role("Custom_role", "A", inplace=True)
    >>> G.get_role_dict()
    {'latents': ['D']}
    """

    SUPPORTED_EDGE_TYPES = ["--", "-o", "o-", "->", "<-", "o>", "<o", "<>", "oo"]

    def __init__(
        self,
        ebunch: Iterable[
            Union[
                tuple[Hashable, Hashable, Hashable],
                tuple[Hashable, Hashable, Hashable, Hashable],
            ]
        ] = None,
        exposures: set[Hashable] = set(),
        outcomes: set[Hashable] = set(),
        latents: set[Hashable] = set(),
        roles=None,
    ):
        """
        Notes
        --------
        - Sub-graph classes must implement `SUPPORTED_EDGE_TYPES`
        """
        super().__init__()
        if ebunch:
            self.add_edges_from(ebunch)

        self.exposures = set(exposures)
        self.outcomes = set(outcomes)
        self.latents = set(latents)

        if roles is None:
            roles = {}
        elif not isinstance(roles, dict):
            raise TypeError("Roles must be provided as a dictionary.")

        # set the roles to the vertices as networkx attributes
        for role, vars in roles.items():
            self.with_role(role=role, variables=vars, inplace=True)

    # ----------------------------------------------------------------------
    # Public API (or Public Methods)
    # ----------------------------------------------------------------------

    def add_edge(
        self,
        u: Hashable,
        v: Hashable,
        type: str = None,
        key: Optional[Hashable] = None,
        **kwargs,
    ):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u, v : node
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        type : str
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

        key : hashable identifier, optional (default=lowest unused integer)
            Used to distinguish multiedges between a pair of nodes.

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G = _CoreGraph()
        >>> G.SUPPORTED_EDGE_TYPES  # check the available edge types
        ['--', '-o', 'o-', '->', '<-', 'o>', '<o', '<>', 'oo']
        >>> G.add_edge("A", "B", "->")
        >>> G.edges  # You can check the key value of the two nodes and the edge connecting the two nodes.
        MultiEdgeView([('A', 'B', 0)])
        >>> G.edges(
        ...     data=True
        ... )  # You can check the type value of the two nodes and the edge connecting the two nodes.
        MultiEdgeDataView([('A', 'B', {'type': '->'})])
        """
        _ = self._validating_and_formatting_edges_value(ebunch=[(u, v, type, key)])
        super().add_edge(u, v, type=type, key=key, **kwargs)

    def add_edges_from(
        self,
        ebunch: Iterable[
            Union[
                tuple[Hashable, Hashable, Hashable],
                tuple[Hashable, Hashable, Hashable, Hashable],
            ]
        ],
        **kwargs,
    ):
        """
        Add all the edges in ebunch.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the
            graph. The edges can be:

                - 3-tuples (u, v, type)
                - 4-tuples (u, v, type, key) for an edge with data and key

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        >>> G.SUPPORTED_EDGE_TYPES  # check the available edge types
        ['--', '-o', 'o-', '->', '<-', 'o>', '<o', '<>', 'oo']
        >>> G.add_edges_from(ebunch=edges)
        >>> G.edges(data=True)
        MultiEdgeDataView([('A', 'B', {'type': '->'}), ('B', 'C', {'type': '->'})])
        """
        ebunch = self._validating_and_formatting_edges_value(ebunch=ebunch)

        for u, v, type, key in ebunch:
            self.add_edge(u, v, type, key=key, **kwargs)

    def remove_edge(
        self,
        u: Hashable,
        v: Hashable,
        type: str = None,
        key: Optional[Hashable] = None,
        **kwargs,
    ):
        """
        Remove an edge between u and v.

        Parameters
        ----------
        u, v : node
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        type : str
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

        key : hashable identifier, optional (default=lowest unused integer)
            Used to distinguish multiedges between a pair of nodes.

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> G.SUPPORTED_EDGE_TYPES  # check the available edge types
        ['--', '-o', 'o-', '->', '<-', 'o>', '<o', '<>', 'oo']
        >>> G.remove_edge("A", "B", "->")
        >>> G.edges(data=True)
        MultiEdgeDataView([('B', 'C', {'type': '->'}), ('C', 'D', {'type': '--'})])
        """
        _ = self._validating_and_formatting_edges_value(ebunch=[(u, v, type, key)])

        if not self.has_edge(u, v):
            raise ValueError("Edge does not exist.")

        if key is None:
            key_type = self[u][v]
            for k in key_type:
                if type == key_type[k].get("type"):
                    key = k
                    break
        if key is None:
            raise ValueError(f"There is no {type} type edge between {u} and {v}.")

        super().remove_edge(u, v, key=key)

    def remove_edges_from(
        self,
        ebunch: Iterable[
            Union[
                tuple[Hashable, Hashable, Hashable],
                tuple[Hashable, Hashable, Hashable, Hashable],
            ]
        ],
        **kwargs,
    ):
        """
        Remove all the edges in ebunch.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the
            graph. The edges can be:

                - 3-tuples (u, v, type)
                - 4-tuples (u, v, type, key) for an edge with data and key

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> G.SUPPORTED_EDGE_TYPES  # check the available edge types
        ['--', '-o', 'o-', '->', '<-', 'o>', '<o', '<>', 'oo']
        >>> remove_edges = [("B", "C", "->"), ("C", "D", "--")]
        >>> G.remove_edges_from(ebunch=remove_edges)
        >>> G.edges(data=True)
        MultiEdgeDataView([('A', 'B', {'type': '->'})])
        """
        ebunch = self._validating_and_formatting_edges_value(ebunch=ebunch)
        for u, v, type, key in ebunch:
            self.remove_edge(u, v, type, key=key, **kwargs)

    def copy(self):
        """
        Returns a copy of the graph object.

        Parameters
        ----------
        None

        Returns
        -------
        graph: graph object
            A copy of the graph object.

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G1 = _CoreGraph()
        >>> G2 = G1.copy()
        >>> G2.__class__
        pgmpy.base.base._CoreGraph

        Notes
        --------
        - This method is expected to be usable without being implemented in a subclass of the graph class.
        """
        ebunch = self._get_edge_type_key()

        graph_copy = self.__class__()
        graph_copy.add_edges_from(ebunch=ebunch)
        graph_copy.exposures = self.exposures
        graph_copy.outcomes = self.outcomes
        graph_copy.latents = self.latents
        for role, vars in self.get_role_dict().items():
            graph_copy.with_role(role=role, variables=vars, inplace=True)

        if not self.__eq__(graph_copy):
            raise ValueError("The graph `copy()` method is not performed correctly.")
        return graph_copy

    # ----------------------------------------------------------------------
    # Internal Methods (or Private Methods)
    # ----------------------------------------------------------------------

    def __eq__(self, other):
        """
        Checks if two graphs are equal. Two graphs are considered equal if they
        have the same nodes, edges, exposures, outcomes, latent variables, and variable roles.

        Parameters
        ----------
        other: graph object
            The other graph to compare with.

        Returns
        -------
        bool:
            True if the graphs are equal, False otherwise.

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G1 = _CoreGraph()
        >>> G2 = _CoreGraph()
        >>> G1.__eq__(G2)
        True

        Notes
        --------
        - This method is expected to be usable without being implemented in a subclass of the graph class.
        """
        if not isinstance(other, self.__class__):
            return False

        self_ebunch = self._get_edge_type_key()
        other_ebunch = other._get_edge_type_key()

        return (
            set(self.nodes()) == set(other.nodes())
            and set(self_ebunch) == set(other_ebunch)
            and self.exposures == other.exposures
            and self.outcomes == other.outcomes
            and self.latents == other.latents
            and self.get_role_dict() == other.get_role_dict()
        )

    def _validating_and_formatting_edges_value(
        self,
        ebunch: Iterable[
            Union[
                tuple[Hashable, Hashable, Hashable],
                tuple[Hashable, Hashable, Hashable, Hashable],
            ]
        ],
    ):
        """
        Helper method that validates the input for
            `add_edges_from()`,
            `remove_edges_from()`.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the
            graph. The edges can be:

                - 3-tuples (u, v, type)
                - 4-tuples (u, v, type, key) for an edge with data and key

        Returns
        -------
        ebunch : list of tuples
            [(`u`, `v`, `type`, `key`), (`u`, `v`, `type`, `key`), ...]
        """
        result = []
        for edge_type_key in ebunch:
            if len(edge_type_key) == 4:
                u, v, type, key = edge_type_key
            else:
                u, v, type = edge_type_key
                key = None
            if (u is None) or (v is None):
                raise ValueError("Nodes cannot be None.")
            if u == v:
                raise ValueError("Nodes cannot be the same for an edge.")
            if (type is None) or (type not in self.SUPPORTED_EDGE_TYPES):
                raise ValueError(f"Types must be one of {self.SUPPORTED_EDGE_TYPES}.")
            result.append((u, v, type, key))
        return result

    def _get_edge_type_key(self):
        """
        Returns the edge's `type` and `key` value connecting the two nodes as a list of tuples.

        Helper method for
            `copy()`,
            `__eq__()`,


        Parameters
        ----------
        None

        Returns
        -------
        ebunch : list of tuples
            [(`u`, `v`, `type`, `key`), (`u`, `v`, `type`, `key`), ...]

        Notes
        --------
        - I expect this method to be useful for creating a graph edge view method.
        """
        return [
            (u, v, data.get("type"), key)
            for u, v, key, data in self.edges(data=True, keys=True)
        ]
