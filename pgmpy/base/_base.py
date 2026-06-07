from collections import deque
from collections.abc import Hashable, Iterable
from typing import Any

import networkx as nx
import pandas as pd

from pgmpy.base._mixin_algorithms import _GraphAlgorithmMixin
from pgmpy.base._mixin_roles import _GraphRolesMixin


class _CoreGraph(nx.MultiGraph, _GraphAlgorithmMixin, _GraphRolesMixin):
    """
    Base class for all graph types in pgmpy.

    This class provides a generalized representation for all graph `edge_types` in pgmpy. All specific graph classes
    (e.g., `DAG`, `PDAG`, ...) inherit from `_CoreGraph`. Subclasses are expected to define their own
    `SUPPORTED_EDGE_TYPES` to restrict the kinds of edges they can store. It also provides generalized algorithms and
    methods that work across all inheriting graphs.

    Parameters
    ----------
    edge_list : iterable of tuples, optional
        A list or iterable of edges of the form [(variable1, variable2, edge_type), ... ] to add at initialization.
        The edge type must be one of the following: "--", "-o", "o-", "->", "<-", "o>", "<o", "<>", "oo"

    latents : set of nodes, (default=set())
        A set of latent variables in the graph. These represent the variables for which we do not have any data.

    exposures : set, (default=set())
        Set of exposure variables in the graph. These are the variables that represent the treatment or intervention
        being studied in a causal analysis. Default is an empty set.

    outcomes : set, (default=set())
        Set of outcome variables in the graph. These are the variables that represent the response or dependent
        variables being studied in a causal analysis. Default is an empty set.

    roles : dict, optional (default=None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).

    Notes
    -----
    **Internal edge representation.**

    Although ``_CoreGraph`` exposes edges through string ``edge_type`` codes such as ``"->"`` or ``"<>"``, it does *not*
    store them as directed edges. Internally the class subclasses an *undirected* ``networkx.MultiGraph``, and the
    orientation of every edge is encoded by a pair of per-endpoint **markers** stored on the edge's data dictionary,
    keyed by the two node identifiers.

    Each endpoint carries one of three markers: ``"-"`` (a tail / no arrowhead), ``">"`` (an arrowhead), or ``"o"`` (a
    circle endpoint whose orientation is unspecified, as used in PAGs). An ``edge_type`` is therefore just the pair of
    endpoint markers: for an edge between ``u`` and ``v`` the dict ``{u: <u-marker>, v: <v-marker>}`` is stored on the
    edge. For example ``"A->B"`` is stored as ``{"A": "-", "B": ">"}`` and ``"A<>B"`` as ``{"A": ">", "B": ">"}``. The
    nine supported codes map to the following ``{u, v}`` marker dicts::

        --  : {u: "-", v: "-"}      ->  : {u: "-", v: ">"}      <-  : {u: ">", v: "-"}
        <>  : {u: ">", v: ">"}      o-  : {u: "o", v: "-"}      -o  : {u: "-", v: "o"}
        o>  : {u: "o", v: ">"}      <o  : {u: ">", v: "o"}      oo  : {u: "o", v: "o"}

    ``_to_markers`` performs the ``edge_type`` to marker-dict conversion and ``_to_edge_type`` the inverse;
    ``get_edges`` / ``get_edge`` use the latter to present edges back as string codes.

    **Why this scheme.**

    The obvious choice, a ``networkx.DiGraph``, can only express a single directed orientation per ordered pair and
    cannot record undirected (``"--"``), bidirected (``"<>"``), or circle (``"o"``) endpoints. Encoding orientation as
    endpoint markers on an undirected graph instead lets a single structure represent every edge type used across
    pgmpy's ``_CoreGraph``-based classes (``PDAG``, ``ADMG``, ``MAG``, ...), each of which restricts itself to a subset
    via ``SUPPORTED_EDGE_TYPES``. Using a *multi*-graph additionally allows a pair of nodes to be joined by more than
    one edge of different types at once (for example a directed ``"A->B"`` together with a bidirected ``"A<>B"``, as
    needed by ADMGs), which a simple graph could not hold. A second edge of the *same* type between a pair is still
    rejected (see ``add_edge``), so the multigraph capacity is used only for genuinely distinct edge types.

    Examples
    --------
    Create an empty `_CoreGraph` with no nodes and no edges.

    >>> from pgmpy.base._base import _CoreGraph
    >>> G = _CoreGraph()

    Edges and vertices can be passed to the constructor as an edge list.

    >>> edges = [("A", "B", "->"), ("B", "C", "->")]
    >>> G = _CoreGraph(edge_list=edges)
    >>> G.get_edges(data=True)
    [('A', 'B', '->'), ('B', 'C', '->')]

    **Nodes:**

    Add one node,

    >>> from pgmpy.base._base import _CoreGraph
    >>> G = _CoreGraph()
    >>> G.add_node("A")
    >>> G.nodes
    NodeView(('A',))

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> from pgmpy.base._base import _CoreGraph
    >>> G = _CoreGraph()
    >>> G.add_edge("A", "B", "->")
    >>> G.get_edges(data=True)
    [('A', 'B', '->')]

    Remove one edge,

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
    >>> G = _CoreGraph(edge_list=edges)
    >>> G.remove_edge("A", "B", "->")
    >>> G.get_edges(data=True)
    [('B', 'C', '->'), ('C', 'D', '--')]

    **Exposures, Outcomes, and Latents:**

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("D", "C", "-o")]
    >>> G = _CoreGraph(edge_list=edges)

    **Roles:**

    Add node's role.

    >>> G.exposures = "A"
    >>> G.outcomes = "C"
    >>> G.latents = "D"

    Checks for the node's role.

    >>> G.exposures
    {'A'}
    >>> G.outcomes
    {'C'}
    >>> G.latents
    {'D'}

    In addition to 'exposures', 'outcomes', and 'latents', you can add custom roles.

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("D", "C", "-o")]
    >>> G = _CoreGraph(edge_list=edges)
    >>> G = G.with_role("Custom_role", "A", inplace=False)
    >>> G = G.with_role("latents", "D", inplace=False)
    >>> G.get_role_dict() == {"latents": ["D"], "Custom_role": ["A"]}
    True
    >>> G = G.without_role("Custom_role", "A", inplace=False)
    >>> G.get_role_dict() == {"latents": ["D"]}
    True

    """

    SUPPORTED_EDGE_TYPES = frozenset(["--", "-o", "o-", "->", "<-", "o>", "<o", "<>", "oo"])

    def __init__(
        self,
        edge_list: Iterable[tuple[Hashable, Hashable, str]] | None = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        latents: set[Hashable] | None = None,
        roles: dict[str, str | Iterable[Hashable]] | None = None,
    ):
        super().__init__()
        if edge_list:
            self._validate_edges(edge_list=edge_list)
            for u, v, edge_type in edge_list:
                self.add_edge(u, v, edge_type=edge_type)

        self.exposures = set() if exposures is None else set(exposures)
        self.outcomes = set() if outcomes is None else set(outcomes)
        self.latents = set() if latents is None else set(latents)

        if roles is None:
            roles = {}
        elif not isinstance(roles, dict):
            raise TypeError("Roles must be provided as a dictionary.")

        # set the roles to the vertices as networkx attributes
        for role, vars in roles.items():
            self.with_role(role=role, variables=vars, inplace=True)

    def add_edge(
        self,
        u: Hashable,
        v: Hashable,
        edge_type: str,
        **kwargs,
    ) -> None:
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u, v : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str
            Must be one of the values in `SUPPORTED_EDGE_TYPES`. This argument is required.

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        None

        See Also
        --------
        add_edges_from : Add multiple edges at once.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph()
        >>> G.add_edge("A", "B", "->")
        >>> G.get_edges(data=True)
        [('A', 'B', '->')]

        """
        self._validate_edges(edge_list=[(u, v, edge_type)])

        # A pair of nodes can have at most one edge of a given type.
        if self.has_edge(u, v, edge_type):
            raise ValueError(f"Edge ({u}, {v}) of type '{edge_type}' already exists in {self.__class__.__name__}.")

        # Directed edges must not close a directed cycle.
        if edge_type == "->" and self.has_node(u) and self.has_node(v) and self.has_direct_path(v, u):
            raise ValueError(f"Adding edge ({u}, {v}, '{edge_type}') would create a directed cycle.")
        if edge_type == "<-" and self.has_node(u) and self.has_node(v) and self.has_direct_path(u, v):
            raise ValueError(f"Adding edge ({u}, {v}, '{edge_type}') would create a directed cycle.")

        markers = self._to_markers(edge=(u, v, edge_type))
        key = super().add_edge(u, v, **kwargs)
        self.edges[u, v, key].update({u: markers[u], v: markers[v]})

    def add_edges_from(
        self,
        edge_list: Iterable[tuple[Hashable, Hashable, str]],
        **kwargs,
    ) -> None:
        """
        Add all the edges in edge_list.

        Parameters
        ----------
        edge_list : list of tuples
            [(`u`, `v`, `edge_type`), (`u`, `v`, `edge_type`), ...]

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        None

        See Also
        --------
        add_edge : Add a single edge.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        >>> G.add_edges_from(edge_list=edges)
        >>> G.get_edges(data=True)
        [('A', 'B', '->'), ('B', 'C', '->')]

        """
        self._validate_edges(edge_list=edge_list)
        for u, v, edge_type in edge_list:
            self.add_edge(u, v, edge_type)

    def remove_edge(
        self,
        u: Hashable,
        v: Hashable,
        edge_type: str,
    ) -> None:
        """
        Remove an edge between u and v.

        Parameters
        ----------
        u, v : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str
            One of the values in `SUPPORTED_EDGE_TYPES`. The edge of this type
            between `u` and `v` is removed. This argument is required.

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        None

        See Also
        --------
        remove_edges_from : Remove multiple edges at once.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> G.remove_edge("A", "B", "->")
        >>> G.get_edges(data=True)
        [('B', 'C', '->'), ('C', 'D', '--')]

        """
        self._validate_edges(edge_list=[(u, v, edge_type)])

        markers = self._to_markers(edge=(u, v, edge_type))
        for key, data in self.get_edge_data(u, v).items():
            if data[u] == markers[u] and data[v] == markers[v]:
                super().remove_edge(u, v, key=key)
                return
        raise ValueError(f"Edge ({u}, {v}, {edge_type}) not in graph.")

    def remove_edges_from(
        self,
        edge_list: Iterable[tuple[Hashable, Hashable, str]],
    ) -> None:
        """
        Remove all the edges in edge_list.

        Parameters
        ----------
        edge_list : list of tuples
            [(`u`, `v`, `edge_type`), (`u`, `v`, `edge_type`), ...]

        Returns
        -------
        None

        See Also
        --------
        remove_edge : Remove a single edge.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> remove_edges = [("B", "C", "->"), ("C", "D", "--")]
        >>> G.remove_edges_from(edge_list=remove_edges)
        >>> G.get_edges(data=True)
        [('A', 'B', '->')]

        """
        self._validate_edges(edge_list=edge_list)
        for u, v, edge_type in edge_list:
            self.remove_edge(u, v, edge_type)

    def copy(self):
        """
        Returns a deep copy of the graph object.

        Parameters
        ----------
        None

        Returns
        -------
        graph: graph object
            A copy of the graph object.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G1 = _CoreGraph()
        >>> G2 = G1.copy()
        >>> G2.__class__
        <class 'pgmpy.base._base._CoreGraph'>

        """
        edge_list = [(u, v, markers) for u, v, markers in self.edges(data=True)]

        graph_copy = self.__class__()
        graph_copy.add_nodes_from(self.nodes(data=True))
        for u, v, markers in edge_list:
            edge_type = self._to_edge_type(u, v, markers=markers)
            graph_copy.add_edge(u, v, edge_type=edge_type)
        for role, vars in self.get_role_dict().items():
            graph_copy.with_role(role=role, variables=vars, inplace=True)

        return graph_copy

    def get_neighbors(self, node: Hashable, edge_type: str | None = None) -> set[Hashable]:
        """
        Returns a set of neighbors nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str (default=None)
            The type should be None or a value from SUPPORTED_EDGE_TYPES.

        Returns
        -------
        nodes : set
            Set of neighbors nodes.

        See Also
        --------
        get_parents : Parent nodes (via incoming directed edges).
        get_children : Child nodes (via outgoing directed edges).
        get_ancestors : All ancestors of a node (including the node itself).
        get_descendants : All descendants of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_neighbors("B", "->")))
        ['C']
        >>> print(sorted(G.get_neighbors("B", "<-")))
        ['A']

        """
        if node not in self.nodes():
            raise ValueError(f"Node {node} not in graph.")

        if (edge_type is not None) and (edge_type not in self.SUPPORTED_EDGE_TYPES):
            raise ValueError(f"Types must be one of {self.SUPPORTED_EDGE_TYPES}. Got {edge_type}.")

        neighboring_nodes = self.neighbors(node)

        if edge_type is None:
            return set(neighboring_nodes)

        filtered_neighbors = set()
        for neighbor in neighboring_nodes:
            edge_data = self.get_edge_data(node, neighbor)
            _markers_dict = self._to_markers(edge=(node, neighbor, edge_type))
            for _, data in edge_data.items():
                if data[node] == _markers_dict[node] and data[neighbor] == _markers_dict[neighbor]:
                    filtered_neighbors.add(neighbor)
                    break

        return filtered_neighbors

    def get_parents(self, node: Hashable) -> set[Hashable]:
        """
        Returns a set of parents nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        nodes : set
            Set of parents nodes.

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_children : Child nodes (via outgoing directed edges).
        get_ancestors : All ancestors of a node (including the node itself).
        get_descendants : All descendants of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_parents("B")))
        ['A']
        >>> print(sorted(G.get_parents("C")))
        ['B']

        """
        return self.get_neighbors(node=node, edge_type="<-")

    def get_children(self, node: Hashable) -> set[Hashable]:
        """
        Returns a set of children nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        nodes : set
            Set of children nodes.

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_parents : Parent nodes (via incoming directed edges).
        get_ancestors : All ancestors of a node (including the node itself).
        get_descendants : All descendants of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_children("A")))
        ['B']
        >>> print(sorted(G.get_children("B")))
        ['C']

        """
        return self.get_neighbors(node=node, edge_type="->")

    def get_spouses(self, node: Hashable) -> set[Hashable]:
        """
        Returns a set of spouses nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        nodes : set
            Set of spouses nodes.

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_parents : Parent nodes (via incoming directed edges).
        get_children : Child nodes (via outgoing directed edges).
        get_descendants : All descendants of a node (including the node itself).
        get_ancestors : All ancestors of a node (including the node itself).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "<>")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_spouses("B")))
        ['C']
        >>> print(sorted(G.get_spouses("C")))
        ['B']

        """
        return self.get_neighbors(node=node, edge_type="<>")

    def get_ancestors(self, node: Hashable) -> set[Hashable]:
        """
        Returns a set of ancestors nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        nodes : set
            Set of ancestors nodes.

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_parents : Parent nodes (via incoming directed edges).
        get_children : Child nodes (via outgoing directed edges).
        get_descendants : All descendants of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_ancestors("C")))
        ['A', 'B', 'C']
        >>> print(sorted(G.get_ancestors("B")))
        ['A', 'B']

        """
        if node not in self.nodes():
            raise ValueError(f"Node {node} not in graph.")

        ancestors = set()
        queue = deque([node])

        while queue:
            current = queue.popleft()
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(self.get_parents(current))
        return ancestors

    def get_descendants(self, node: Hashable) -> set[Hashable]:
        """
        Returns a set of descendants nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        nodes : set
            Set of descendants nodes.

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_parents : Parent nodes (via incoming directed edges).
        get_children : Child nodes (via outgoing directed edges).
        get_ancestors : All ancestors of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_reachable_nodes : Nodes reachable from a node via a given edge type.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_descendants("A")))
        ['A', 'B', 'C']
        >>> print(sorted(G.get_descendants("B")))
        ['B', 'C']

        """
        if node not in self.nodes():
            raise ValueError(f"Node {node} not in graph.")

        descendants = set()
        queue = deque([node])

        while queue:
            current = queue.popleft()
            if current not in descendants:
                descendants.add(current)
                queue.extend(self.get_children(current))
        return descendants

    def get_reachable_nodes(self, node: Hashable, edge_type: str | None = None) -> set[Hashable]:
        """
        Returns a set of reachable nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

        Returns
        -------
        nodes : set
            Set of reachable nodes.

        See Also
        --------
        get_neighbors : Adjacent nodes, optionally filtered by edge type.
        get_parents : Parent nodes (via incoming directed edges).
        get_children : Child nodes (via outgoing directed edges).
        get_ancestors : All ancestors of a node (including the node itself).
        get_spouses : Spouse nodes (via bidirected edges).
        get_descendants : All descendants of a node (including the node itself).

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [
        ...     ("A", "B", "->"),
        ...     ("B", "C", "->"),
        ...     ("C", "D", "--"),
        ...     ("D", "F", "<>"),
        ... ]
        >>> G = _CoreGraph(edge_list=edges)
        >>> print(sorted(G.get_reachable_nodes("A", "->")))
        ['A', 'B', 'C']
        >>> print(sorted(G.get_reachable_nodes("C", "--")))
        ['C', 'D']
        >>> print(sorted(G.get_reachable_nodes("D", "<>")))
        ['D', 'F']

        """
        if node not in self.nodes():
            raise ValueError(f"Node {node} not in graph.")

        reachable = set()
        queue = deque([node])

        while queue:
            current = queue.popleft()
            if current not in reachable:
                reachable.add(current)
                queue.extend(self.get_neighbors(current, edge_type=edge_type))
        return reachable

    def get_edges(self, data: bool = True) -> list[tuple[Any, ...]]:
        """
        Retrieve edges with optional API-formatted edge types.

        Parameters
        ----------
        data : bool, optional (default=True)
            If True, returns the edge type as a string (e.g., '->') instead of
            the internal dictionary representation. Default is True.

        Returns
        -------
        list
            A list of edge tuples. The format varies based on parameters:
            * (u, v, type)      : data=True
            * (u, v)            : data=False

        See Also
        --------
        get_edge : Edges between a specific pair of nodes.

        Examples
        --------
        >>> edges = [("A", "B", "->"), ("A", "B", "<>"), ("B", "C", "->")]
        >>> G = _CoreGraph(edge_list=edges)
        >>> G.get_edges(data=True)
        [('A', 'B', '->'), ('A', 'B', '<>'), ('B', 'C', '->')]
        >>> G.get_edges(data=False)
        [('A', 'B'), ('A', 'B'), ('B', 'C')]

        """
        networkx_edge_list = super().edges(data=data)

        # (u, v)
        if data is False:
            return list(networkx_edge_list)

        # (u, v, edge_type)
        return [(u, v, self._to_edge_type(u, v, data)) for u, v, data in networkx_edge_list]

    def get_edge(self, u: Hashable, v: Hashable, data: bool = True) -> list[tuple[Hashable, ...]]:
        """
        Retrieve edge with optional API-formatted edge types.

        Parameters
        ----------
        u : Hashable
            The source node of the edge.
        v : Hashable
            The target node of the edge.

        Returns
        -------
        edge : tuple
            edge tuples.
            * (u, v, type)      : data=True
            * (u, v)            : data=False

        data : bool, optional (default=True)
            If True, returns the edge type as a string (e.g., '->') instead of
            the internal dictionary representation. Default is True.

        See Also
        --------
        get_edges : All edges in the graph.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph()
        >>> graph.add_edge("A", "B", "->")
        >>> graph.add_edge("A", "B", "<>")
        >>> graph.add_edge("B", "C", "--")
        >>> set(graph.get_edge("A", "B"))
        {('A', 'B', '->'), ('A', 'B', '<>')}
        >>> set(graph.get_edge("B", "C"))
        {('B', 'C', '--')}

        """
        result: list[tuple[Hashable, ...]] = []
        if not self.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) not in graph.")

        keys = self[u][v]
        for key_val, marker in keys.items():
            if data:
                edge_type = self._to_edge_type(u, v, marker)
                result.append((u, v, edge_type))
            else:
                result.append((u, v))
        return result

    def has_edge(self, u, v, edge_type=None):
        """
        Returns True if the graph has an edge between nodes u and v.

        Parameters
        ----------
        u : Hashable
            The source node of the edge.
        v : Hashable
            The target node of the edge.
        edge_type : str
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

        Returns
        -------
        bool

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph()
        >>> graph.add_edge("A", "B", "->")
        >>> graph.has_edge("A", "B")
        True
        >>> graph.has_edge("A", "B", "->")
        True
        >>> graph.has_edge("A", "B", "--")
        False

        """
        if not super().has_edge(u, v):
            return False

        if edge_type is None:
            return True

        if edge_type not in self.SUPPORTED_EDGE_TYPES:
            raise ValueError(f"Types must be one of {self.SUPPORTED_EDGE_TYPES}.")
        edge_list = self.get_edge(u, v)

        for edge in edge_list:
            if edge[2] == edge_type:
                return True
        return False

    def replace_edge(
        self,
        u: Hashable,
        v: Hashable,
        old_type: str = "--",
        new_type: str = "->",
    ) -> None:
        """
        Replaces the type of an existing edge between two nodes with a new type.

        Parameters
        ----------
        u : Hashable
            The source node of the edge.
        v : Hashable
            The target node of the edge.
        old_type : str, optional (default="--")
            The current type of the edge that needs to be replaced.
        new_type : str, optional (default="->")
            The new edge type to be assigned.

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph()
        >>> graph.add_edge("A", "B", "--")
        >>> graph.replace_edge("A", "B", old_type="--", new_type="->")
        >>> graph.has_edge("A", "B", "->")
        True
        >>> graph.has_edge("A", "B", "--")
        False

        """
        if (old_type not in self.SUPPORTED_EDGE_TYPES) or (new_type not in self.SUPPORTED_EDGE_TYPES):
            raise ValueError(
                f"Unsupported edge type(s) provided. "
                f"Got old_type='{old_type}', new_type='{new_type}'. "
                f"Supported types are: {self.SUPPORTED_EDGE_TYPES}."
            )
        if not self.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) not in graph.")

        # TODO: Add logic of maintain edge's attr data
        #       Currently, we treat `edge_type` as an attribute,
        #       so additional logic will need to be developed to handle this in the future.
        self.remove_edge(u, v, old_type)
        self.add_edge(u, v, new_type)

    def to_adjacency(self, encoding: str = "edge_type", nodelist=None) -> pd.DataFrame:
        """
        Return the adjacency matrix of the graph as a ``pandas.DataFrame``.

        The value in each cell depends on ``encoding``, chosen to match the conventions of common ecosystem tools.

        Parameters
        ----------
        encoding : str (default="edge_type")
            One of:

            - ``"edge_type"`` : native, human-readable. ``M.loc[u, v]`` is the edge type oriented from ``u`` to ``v``
              (e.g. ``"->"``, ``"<-"``, ``"<>"``, ``"--"``, ``"o>"``), or ``0`` for no edge. When a pair is joined by
              more than one edge (e.g. an ADMG with a directed *and* a bidirected edge), the cell holds a tuple of edge
              types, aligned per edge with the mirror cell ``M.loc[v, u]``.
            - ``"binary"`` : ``0/1`` directed adjacency. ``M.loc[u, v] = 1`` iff there is a directed edge ``u`` -> ``v``
              (so directed edges are asymmetric); an undirected edge is symmetric (both ``1``). This is the
              DAG/CPDAG-style numeric matrix and is only supported for DAG/PDAG-style graphs, i.e. those whose
              ``SUPPORTED_EDGE_TYPES`` is a subset of ``{"->", "<-", "--"}``. ``"bnlearn"`` is accepted as an alias.
            - ``"causal-learn"`` : integer codes interoperable with causal-learn's ``GeneralGraph``. ``M.loc[u, v]`` is
              the mark at ``u`` (the row endpoint): ``0`` no edge, ``-1`` tail, ``1`` arrowhead, ``2`` circle, ``4``
              tail-and-arrowhead, ``5`` arrowhead-and-arrowhead (the last two for coincident directed + bidirected
              edges).
            - ``"pcalg"`` : integer codes of pcalg's ``amat.pag``. ``M.loc[u, v]`` is the mark at ``v`` (the column
              endpoint): ``0`` no edge, ``1`` circle, ``2`` arrowhead, ``3`` tail.

        nodelist : list, optional (default=None)
            The row/column ordering. If ``None``, the sorted node list is used.

        Returns
        -------
        pandas.DataFrame
            A square adjacency matrix indexed by the nodes.

        Raises
        ------
        ValueError
            If ``encoding`` is unknown; if ``encoding="binary"`` is requested on a graph that supports edge types
            beyond ``{"->", "<-", "--"}`` (i.e. anything other than a DAG/PDAG); or if the graph contains an edge the
            chosen encoding cannot represent (coincident edges for ``"binary"``/``"pcalg"``).

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>")])
        >>> G.to_adjacency().loc["A", "B"]
        '->'
        """
        # Step 0: Validate the inputs
        valid = {"edge_type", "binary", "bnlearn", "causal-learn", "pcalg"}
        if encoding not in valid:
            raise ValueError(f"encoding must be one of {sorted(valid)}. Got {encoding!r}.")
        if encoding == "bnlearn":
            # binary and bnlearn are equivalent
            encoding = "binary"
        if encoding == "binary" and not self.SUPPORTED_EDGE_TYPES <= {"->", "<-", "--"}:
            raise ValueError(
                "binary encoding is only supported for DAG and PDAG (directed/undirected edges); "
                f"{self.__class__.__name__} supports {sorted(self.SUPPORTED_EDGE_TYPES)}."
            )

        # Step 1: Define data structure and helper method required for constructing the adjacency matrix.
        nodes = sorted(self.nodes()) if nodelist is None else list(nodelist)
        adj = pd.DataFrame(0, index=nodes, columns=nodes, dtype=object if encoding == "edge_type" else int)

        pcalg_code = {("o",): 1, (">",): 2, ("-",): 3}
        causal_learn_code = {("-",): -1, (">",): 1, ("o",): 2, ("-", ">"): 4, (">", ">"): 5}

        def endpoint_code(code_map, marks, u, v):
            """
            Used for converting edge_type adjacency matrix to pcalg or causal_learn conventions.
            """
            key = tuple(sorted(marks))
            if key not in code_map:
                raise ValueError(f"{encoding} adjacency cannot represent the marks {key} between {u} and {v}.")
            return code_map[key]

        # Step 2: Iterate over edges and add the appropriate symbol in `adj` matrix.
        for u, v in dict.fromkeys(self.get_edges(data=False)):
            edges = list(self.get_edge_data(u, v).values())
            marks_u = [data[u] for data in edges]
            marks_v = [data[v] for data in edges]

            # Step 2.1: If encoding = "edge_type", add the edge type at both (u, v) and (v, u). Use a tuple if there are
            #           multiple edges between u and v.
            if encoding == "edge_type":
                if len(edges) == 1:
                    adj.at[u, v] = self._to_edge_type(u, v, edges[0])
                    adj.at[v, u] = self._to_edge_type(v, u, edges[0])
                else:
                    order = sorted(range(len(edges)), key=lambda i: (marks_u[i], marks_v[i]))
                    adj.at[u, v] = tuple(self._to_edge_type(u, v, edges[i]) for i in order)
                    adj.at[v, u] = tuple(self._to_edge_type(v, u, edges[i]) for i in order)

            # Step 2.2: If encoding = "binary", add directed and undirected edges.
            elif encoding == "binary":
                if len(edges) > 1:
                    raise ValueError(
                        f"binary adjacency cannot represent the {len(edges)} coincident edges between {u} and {v}."
                    )
                edge_type = self._to_edge_type(u, v, edges[0])
                if edge_type == "->":
                    adj.at[u, v] = 1
                elif edge_type == "<-":
                    adj.at[v, u] = 1
                else:
                    adj.at[u, v] = 1
                    adj.at[v, u] = 1

            # Step 2.3: If encoding = "pcalg", convert the edge type using `pcalg_code` dict.
            elif encoding == "pcalg":
                adj.at[u, v] = endpoint_code(pcalg_code, marks_v, u, v)
                adj.at[v, u] = endpoint_code(pcalg_code, marks_u, u, v)

            # Step 2.4: If encoding = "causal-learn", convert the edge type using `causal_learn_code` dict.
            else:
                adj.at[u, v] = endpoint_code(causal_learn_code, marks_u, u, v)
                adj.at[v, u] = endpoint_code(causal_learn_code, marks_v, u, v)

        # Step 3: Return the dataframe.
        return adj

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
        >>> from pgmpy.base._base import _CoreGraph
        >>> G1 = _CoreGraph()
        >>> G2 = _CoreGraph()
        >>> G1.__eq__(G2)
        True

        """
        if not isinstance(other, self.__class__):
            return False
        return nx.utils.graphs_equal(self, other) and self.get_role_dict() == other.get_role_dict()

    def _validate_edges(
        self,
        edge_list,
    ):
        """
        Validates the value input by the user, then either raises an error.

        Parameters
        ----------
        edge_list : list of tuples
            [(`u`, `v`, `edge_type`), (`u`, `v`, `edge_type`), ...]
        """
        for edge in edge_list:
            if len(edge) != 3:
                raise ValueError(f"Edge tuple must have 3 elements. Edge {edge} is of length {len(edge)}.")

            u, v, edge_type = edge

            if (u is None) or (v is None):
                raise ValueError(f"Nodes cannot be None. Got {(u, v, edge_type)}.")
            if u == v:
                raise ValueError(f"Nodes cannot be the same for an edge. Got {(u, v, edge_type)}.")
            if not isinstance(edge_type, str) or edge_type not in self.SUPPORTED_EDGE_TYPES:
                raise ValueError(f"edge_type must be one of {self.SUPPORTED_EDGE_TYPES}. Got {(u, v, edge_type)}.")

    def _to_markers(
        self,
        edge: tuple[Hashable, Hashable, str],
    ) -> dict[Hashable, str]:
        """
        The `_to_markers` method converts the user's `edge_type` input into an internal representation.
        """
        u, v, edge_type = edge
        if edge_type == "<-":
            return {v: "-", u: ">"}
        elif edge_type == "o-":
            return {v: "-", u: "o"}
        elif edge_type == "<o":
            return {v: "o", u: ">"}
        elif edge_type == "<>":
            return {u: ">", v: ">"}
        else:
            return {u: edge_type[0], v: edge_type[1]}

    def _to_edge_type(
        self,
        u: Hashable,
        v: Hashable,
        markers: dict,
    ) -> str:
        """
        The `_to_edge_type` method converts the internal representation into the user's `edge_type` input.
        """
        u_marker = markers[u]
        v_marker = markers[v]

        marker_map = {
            (">", "-"): "<-",
            ("o", "-"): "o-",
            (">", "o"): "<o",
            (">", ">"): "<>",
        }
        return marker_map.get((u_marker, v_marker), f"{u_marker}{v_marker}")
