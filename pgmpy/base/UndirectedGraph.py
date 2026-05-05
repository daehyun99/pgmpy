#!/usr/bin/env python3
"""Base class for undirected graphical models in pgmpy."""

import itertools

import networkx as nx

from pgmpy.base._base import _CoreGraph


class UndirectedGraph(_CoreGraph):
    """Base class for all the Undirected Graphical models.

    Each node in the graph can represent either a random variable, `Factor`,
    or a cluster of random variables. Edges in the graph are interactions
    between the nodes.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data can be an edge list or any Networkx graph object.

    Examples
    --------
    Create an empty UndirectedGraph with no nodes and no edges

    >>> from pgmpy.base import UndirectedGraph
    >>> G = UndirectedGraph()

    G can be grown in several ways

    **Nodes:**

    Add one node at a time:

    >>> G.add_node("a")

    Add the nodes from any container (a list, set or tuple or the nodes
    from another graph).

    >>> G.add_nodes_from(["a", "b"])

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge("a", "b")

    a list of edges,

    >>> G.add_edges_from([("a", "b"), ("b", "c")])

    If some edges connect nodes not yet in the model, the nodes
    are added automatically.  There are no errors when adding
    nodes or edges that already exist.

    **Shortcuts:**

    Many common graph features allow python syntax for speed reporting.

    >>> "a" in G  # check if node in graph
    True
    >>> len(G)  # number of nodes in graph
    3

    """

    SUPPORTED_EDGE_TYPES = frozenset(["--"])

    def __init__(self, ebunch=None):
        """Initialize UndirectedGraph with optional edges."""
        super().__init__()
        edge_list = ebunch
        if edge_list:
            self._validate_edges(edge_list=edge_list)
            for edge in edge_list:
                if len(edge) == 3:
                    u, v, edge_type = edge
                elif len(edge) == 2:
                    u, v = edge
                    edge_type = "--"

                self.add_edge(u, v, edge_type=edge_type)

    def add_edge(
        self,
        u,
        v,
        edge_type: str = "--",
        **kwargs,
    ) -> None:
        """Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge.

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(["Alice", "Bob", "Charles"])
        >>> G.add_edge(u="Alice", v="Bob")
        >>> sorted(G.nodes())
        ['Alice', 'Bob', 'Charles']
        >>> sorted(G.edges())
        [('Alice', 'Bob')]

        When the node is not already present in the graph:

        >>> G.add_edge(u="Alice", v="Ankur")
        >>> sorted(G.nodes())
        ['Alice', 'Ankur', 'Bob', 'Charles']
        >>> sorted(G.edges())
        [('Alice', 'Ankur'), ('Alice', 'Bob')]

        Adding edges with weight:

        >>> G.add_edge("Ankur", "Maria", weight=0.1)
        >>> G.edges["Ankur", "Maria"]
        {'weight': 0.1}

        """
        super().add_edge(u, v, edge_type=edge_type)

    def add_edges_from(self, ebunch, weights=None):
        """Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names can be any hashable python
        object.

        **The behavior of adding weights is different than networkx.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the edge at index i.

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(["Alice", "Bob", "Charles"])
        >>> G.add_edges_from(ebunch=[("Alice", "Bob"), ("Bob", "Charles")])
        >>> sorted(G.nodes())
        ['Alice', 'Bob', 'Charles']
        >>> sorted(G.edges())
        [('Alice', 'Bob'), ('Bob', 'Charles')]

        When the node is not already in the model:

        >>> G.add_edges_from(ebunch=[("Alice", "Ankur")])
        >>> sorted(G.nodes())
        ['Alice', 'Ankur', 'Bob', 'Charles']
        >>> sorted(G.edges())
        [('Alice', 'Ankur'), ('Alice', 'Bob'), ('Bob', 'Charles')]

        Adding edges with weights:

        >>> G.add_edges_from(
        ...     [("Ankur", "Maria"), ("Maria", "Mason")], weights=[0.3, 0.5]
        ... )
        >>> G.edges["Ankur", "Maria"]
        {'weight': 0.3}
        >>> G.edges["Maria", "Mason"]
        {'weight': 0.5}

        """
        ebunch = list(ebunch)

        if weights:
            if len(ebunch) != len(weights):
                raise ValueError("The number of elements in ebunch and weightsshould be equal")
            for index in range(len(ebunch)):
                self.add_edge(ebunch[index][0], ebunch[index][1], weight=weights[index])
        else:
            for edge in ebunch:
                self.add_edge(edge[0], edge[1])

    def is_clique(self, nodes):
        """Check if the given nodes form a clique.

        Parameters
        ----------
        nodes: list, array-like
            List of nodes to check if they are a part of any clique.

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph(
        ...     ebunch=[
        ...         ("A", "B"),
        ...         ("C", "B"),
        ...         ("B", "D"),
        ...         ("B", "E"),
        ...         ("D", "E"),
        ...         ("E", "F"),
        ...         ("D", "F"),
        ...         ("B", "F"),
        ...     ]
        ... )
        >>> G.is_clique(nodes=["A", "B", "C", "D"])
        False
        >>> G.is_clique(nodes=["B", "D", "E", "F"])
        True

        Since B, D, E and F are clique, any subset of these should also
        be clique.

        >>> G.is_clique(nodes=["D", "E", "B"])
        True

        """
        for node1, node2 in itertools.combinations(nodes, 2):
            if not self.has_edge(node1, node2):
                return False
        return True

    def is_triangulated(self):
        """Check whether the undirected graph is triangulated (chordal) or not.

        Chordal Graph: A chordal graph is one in which all cycles of four
                       or more vertices have a chord.

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_edges_from(
        ...     ebunch=[("x1", "x2"), ("x1", "x3"), ("x2", "x4"), ("x3", "x4")]
        ... )
        >>> G.is_triangulated()
        False
        >>> G.add_edge(u="x1", v="x4")
        >>> G.is_triangulated()
        True

        """
        return nx.is_chordal(self)

    def is_acyclic(self):
        return True

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

        Notes
        -----
        Helper method that validates the input for
            `add_edge()`,
            `add_edges_from()`,
            `remove_edge()`,
            `remove_edges_from()`.
        """
        if not edge_list:
            return
        supported_types = self.SUPPORTED_EDGE_TYPES

        for edge in edge_list:
            if len(edge) == 3:
                u, v, edge_type = edge
            elif len(edge) == 2:
                u, v = edge
                edge_type = "--"
            else:
                raise ValueError(f"Edge tuple must have 3 elements. Got {len(edge)}.")

            if (u is None) or (v is None):
                raise ValueError("Nodes cannot be None.")
            if u == v:
                raise ValueError("Nodes cannot be the same for an edge.")
            if not isinstance(edge_type, str | None):
                raise ValueError("edge_type must be a string.")
            if edge_type is not None and edge_type not in supported_types:
                raise ValueError(f"Types must be one of {supported_types}.")

    def _validate_graph_specific_edge(self, u, v, edge_type):
        """
        Validates graph-specific constraints on the given edge.

        Parameters
        ----------
        u, v : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str (default="->")
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

        Notes
        -----
        Helper method that validates constraints specific to a graph subclass,
        beyond the common checks performed in `_validate_edges()`.

        Intended to be implemented by subclasses.
        """
        if not self.is_multigraph():
            if self.has_edge(u, v, edge_type):
                self.remove_edge(u, v, edge_type="--")
        if self.is_acyclic():
            if edge_type == "->":
                if self.has_node(u) and self.has_node(v) and self.has_direct_path(v, u):
                    raise ValueError(f"Direct cycles are not allowed in a {self.__class__.__name__}.")
            elif edge_type == "<-":
                if self.has_node(u) and self.has_node(v) and self.has_direct_path(u, v):
                    raise ValueError(f"Direct cycles are not allowed in a {self.__class__.__name__}.")

    def remove_edge(
        self,
        u,
        v,
        edge_type: str = "--",
    ) -> None:
        """
        Remove an edge between u and v.

        Parameters
        ----------
        u, v : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str (default=None)
            The type should be None or a value from SUPPORTED_EDGE_TYPES.
            If the type is `None`, remove all edges between `u` and `v`.

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        None

        See Also
        --------
        `remove_edges_from()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

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

        self._remove_edge(u, v, edge_type)

    def remove_edges_from(
        self,
        edge_list,
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
        `remove_edge()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

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
        for u, v in edge_list:
            self._remove_edge(u, v)
