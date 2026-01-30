from collections import deque
from typing import Hashable, Iterable, Optional

import networkx as nx

from pgmpy.base._mixin_roles import _GraphRolesMixin


class _CoreGraph(nx.MultiGraph, _GraphRolesMixin):
    """
    Base graph class for pgmpy.

    This class provides a generalized representation for all graph edge_types in pgmpy.
    All specific graph classes (e.g., DAG, PDAG, ...) inherit from _CoreGraph.

    Subclasses are expected to define their own `SUPPORTED_EDGE_TYPES` to restrict the kinds of edges they can store.

    It also provides generalized algorithms and methods that work across all inheriting graph edge_types.

    Parameters
    ----------
    ebunch : iterable of tuples, optional
        A list or iterable of edges to add at initialization.

    latents : set of nodes, default=set()
        A set of latent variables in the graph. These are not observed
        variables but are used to represent unobserved confounding or
        other latent structures.

    exposures : set, default=set()
        Set of exposure variables in the graph. These are the variables
        that represent the treatment or intervention being studied in a
        causal analysis. Default is an empty set.

    outcomes : set, default=set()
        Set of outcome variables in the graph. These are the variables
        that represent the response or dependent variables being studied
        in a causal analysis. Default is an empty set.

    roles : dict, optional (default: None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).

    Examples
    --------
    Create an empty `_CoreGraph` with no nodes and no edges.

    >>> from pgmpy.base._base import _CoreGraph
    >>> G = _CoreGraph()

    Edges and vertices can be passed to the constructor as an edge list.

    >>> edges = [("A", "B", "->"), ("B", "C", "->")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> graph.edges(keys=True, data=True)  # [need to fix]
    [('A', 'B', {'edge_type': '->'}), ('B', 'C', {'edge_type': '->'})]

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
    >>> graph.edges(keys=True, data=True)  # [need to fix]
    [('A', 'B', {'edge_type': '->'})]

    Remove one edge,

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> G.remove_edge("A", "B", "->")
    >>> graph.edges(keys=True, data=True)  # [need to fix]
    [('B', 'C', {'edge_type': '->'}), ('C', 'D', {'edge_type': '--'})]

    **Exposures, Outcomes, and Latents:**

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("D", "C", "-o")]
    >>> G = _CoreGraph(ebunch=edges)

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

    In addition to 'exposure', 'outcomes', and 'latents', you can add custom roles.

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("D", "C", "-o")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> G = G.with_role("Custom_role", "A", inplace=False)
    >>> G = G.with_role("latents", "D", inplace=False)
    >>> G.get_role_dict() == {"latents": ["D"], "Custom_role": ["A"]}
    True
    >>> G = G.without_role("Custom_role", "A", inplace=False)
    >>> G.get_role_dict() == {"latents": ["D"]}
    True

    """

    SUPPORTED_EDGE_TYPES = frozenset(
        ["--", "-o", "o-", "->", "<-", "o>", "<o", "<>", "oo"]
    )

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable, Hashable]] = None,
        exposures: Optional[set[Hashable]] = None,
        outcomes: Optional[set[Hashable]] = None,
        latents: Optional[set[Hashable]] = None,
        roles=None,
    ):
        super().__init__()
        if ebunch:
            self.add_edges_from(ebunch)

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

    # ----------------------------------------------------------------------
    # Public API (or Public Methods)
    # ----------------------------------------------------------------------

    def add_edge(
        self,
        u: Hashable,
        v: Hashable,
        edge_type: str = "->",
        key=None,
        **kwargs,
    ):
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
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        key : Hashable

        Returns
        -------
        None

        See Also
        --------
        `add_edges_from()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.
        The edge_type value is matched with the key value of networkx.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph()
        >>> G.add_edge("A", "B", "->")
        >>> graph.edges(keys=True, data=True)  # [need to fix]
        [('A', 'B', {'edge_type': '->'})]

        """
        if isinstance(edge_type, dict):  # Use case for `copy` method
            _, _, edge_type = self._unpreprocess_edge(u, v, edge_type)

        self._validate_edges(ebunch=[(u, v, edge_type)])

        u, v, edge_type = self._preprocess_edge(u, v, edge_type)
        _key = super().add_edge(u, v, key=key, **kwargs)
        self.edges[u, v, _key].update({u: edge_type[0], v: edge_type[1]})

    def add_edges_from(
        self,
        ebunch: (
            Iterable[tuple[Hashable, Hashable, Hashable]]
            | Iterable[tuple[Hashable, Hashable, Hashable, Hashable]]
        ),
        **kwargs,
    ):
        """
        Add all the edges in ebunch.

        Parameters
        ----------
        ebunch : list of tuples
            [(`u`, `v`, `edge_type`), (`u`, `v`, `edge_type`), ...]
            [(`u`, `v`, `key`, `edge_type`), (`u`, `v`, `key`, `edge_type`), ...]

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        None

        See Also
        --------
        `add_edge()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        >>> G.add_edges_from(ebunch=edges)
        >>> graph.edges(keys=True, data=True)  # [need to fix]
        [('A', 'B', {'edge_type': '->'}), ('B', 'C', {'edge_type': '->'})]

        """
        self._validate_edges(ebunch=ebunch)

        for edge in ebunch:
            if len(edge) == 3:
                u, v, edge_type = edge
                self.add_edge(u, v, edge_type=edge_type, **kwargs)
            elif len(edge) == 4:
                u, v, key, edge_type = edge
                self.add_edge(u, v, edge_type=edge_type, key=key, **kwargs)

    def remove_edge(
        self,
        u: Hashable,
        v: Hashable,
        edge_type: str = None,
    ):
        """
        Remove an edge between u and v.

        Parameters
        ----------
        u, v : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

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
        The edge_type value is matched with the key value of networkx.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> G.remove_edge("A", "B", "->")
        >>> graph.edges(keys=True, data=True)  # [need to fix]
        [('B', 'C', {'edge_type': '->'}), ('C', 'D', {'edge_type': '--'})]

        """
        self._validate_edges(ebunch=[(u, v, edge_type)])

        u, v, edge_type = self._preprocess_edge(u, v, edge_type)

        keys_to_remove = []
        edges = self.get_edge_data(u, v)
        if edge_type is None:
            keys_to_remove = list(edges.keys())
        else:
            for key, data in edges.items():
                if data[u] == edge_type[0] and data[v] == edge_type[1]:
                    keys_to_remove.append(key)

        if len(keys_to_remove) == 0:
            raise ValueError(f"Edge ({u}, {v}, {edge_type}) not in graph.")
        else:
            for key in keys_to_remove:
                super().remove_edge(u, v, key=key)

    def remove_edges_from(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable, Hashable]],
    ):
        """
        Remove all the edges in ebunch.

        Parameters
        ----------
        ebunch : list of tuples
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
        >>> G = _CoreGraph(ebunch=edges)
        >>> remove_edges = [("B", "C", "->"), ("C", "D", "--")]
        >>> G.remove_edges_from(ebunch=remove_edges)
        >>> graph.edges(keys=True, data=True)  # [need to fix]
        [('A', 'B', {'edge_type': '->'})]

        """
        self._validate_edges(ebunch=ebunch)

        for u, v, edge_type in ebunch:
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

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G1 = _CoreGraph()
        >>> G2 = G1.copy()
        >>> G2.__class__
        <class 'pgmpy.base._base._CoreGraph'>

        """
        ebunch = [
            (u, v, key, edge_type)
            for u, v, key, edge_type in self.edges(keys=True, data=True)
        ]

        graph_copy = self.__class__()
        graph_copy.add_nodes_from(self.nodes(data=True))
        graph_copy.add_edges_from(ebunch=ebunch)
        for role, vars in self.get_role_dict().items():
            graph_copy.with_role(role=role, variables=vars, inplace=True)

        return graph_copy

    def get_neighbors(self, node, edge_type=None):
        """
        Returns a set of neighbors nodes in the graph.

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
            Set of neighbors nodes.

        See Also
        --------
        `get_parents()`
        `get_children()`
        `get_ancestors()`
        `get_descendants()`
        `get_spouses()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> print(sorted(G.get_neighbors("B", "->")))
        ['C']
        >>> print(sorted(G.get_neighbors("B", "<-")))
        ['A']

        """
        self._validate_nodes(node=node, edge_type=edge_type)

        neighboring_nodes = self.neighbors(node)

        if edge_type is None:
            return set(neighboring_nodes)

        else:
            if edge_type[0] == "<":
                edge_type = f">{edge_type[1]}"

            filtered_neighbors = set()
            for neighbor in neighboring_nodes:
                edge_data = self.get_edge_data(node, neighbor)
                for _, data in edge_data.items():
                    if data[node] == edge_type[0] and data[neighbor] == edge_type[1]:
                        filtered_neighbors.add(neighbor)
                        break

        return filtered_neighbors

    def get_parents(self, node):
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
        `get_neighbors()`
        `get_children()`
        `get_ancestors()`
        `get_descendants()`
        `get_spouses()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> print(sorted(G.get_parents("B")))
        ['A']
        >>> print(sorted(G.get_parents("C")))
        ['B']

        """
        return self.get_neighbors(node=node, edge_type="<-")

    def get_children(self, node):
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
        `get_neighbors()`
        `get_parents()`
        `get_ancestors()`
        `get_descendants()`
        `get_spouses()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> print(sorted(G.get_children("A")))
        ['B']
        >>> print(sorted(G.get_children("B")))
        ['C']

        """
        return self.get_neighbors(node=node, edge_type="->")

    def get_spouses(self, node):
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
        `get_neighbors()`
        `get_parents()`
        `get_children()`
        `get_descendants()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "<>")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> print(sorted(G.get_spouses("B")))
        ['C']
        >>> print(sorted(G.get_spouses("C")))
        ['B']

        """
        return self.get_neighbors(node=node, edge_type="<>")

    def get_ancestors(self, node):
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
        `get_neighbors()`
        `get_parents()`
        `get_children()`
        `get_descendants()`
        `get_spouses()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(ebunch=edges)
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

    def get_descendants(self, node):
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
        `get_neighbors()`
        `get_parents()`
        `get_children()`
        `get_ancestors()`
        `get_spouses()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(ebunch=edges)
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

    def get_reachable_nodes(self, node, edge_type):
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
        `get_neighbors()`
        `get_parents()`
        `get_children()`
        `get_ancestors()`
        `get_spouses()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [
        ...     ("A", "B", "->"),
        ...     ("B", "C", "->"),
        ...     ("C", "D", "--"),
        ...     ("D", "F", "<>"),
        ... ]
        >>> G = _CoreGraph(ebunch=edges)
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

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

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
        return (
            nx.utils.graphs_equal(self, other)
            and self.get_role_dict() == other.get_role_dict()
        )

    def _validate_edges(
        self,
        ebunch: (
            Iterable[tuple[Hashable, Hashable, Hashable]]
            | Iterable[tuple[Hashable, Hashable, Hashable, Hashable]]
        ),
    ):
        """
        Validates the value input by the user, then either raises an error.

        Parameters
        ----------
        ebunch : list of tuples
            [(`u`, `v`, `edge_type`), (`u`, `v`, `edge_type`), ...]

        Notes
        -----
        Helper method that validates the input for
            `add_edge()`,
            `add_edges_from()`,
            `remove_edge()`,
            `remove_edges_from()`.
        """
        if not ebunch:
            return
        supported_types = self.SUPPORTED_EDGE_TYPES

        for edge in ebunch:
            if len(edge) == 3:
                u, v, edge_type = edge
            elif len(edge) == 4:
                u, v, _, edge_type = edge
            else:
                raise ValueError(
                    f"Edge tuple must have 3 or 4 elements. Got {len(edge)}."
                )

            if (u is None) or (v is None):
                raise ValueError("Nodes cannot be None.")
            if u == v:
                raise ValueError("Nodes cannot be the same for an edge.")

            if isinstance(edge_type, dict):
                _, _, edge_type = self._unpreprocess_edge(u, v, edge_type)

            if edge_type is None or edge_type not in supported_types:
                raise ValueError(f"Types must be one of {supported_types}.")

    def _validate_nodes(self, node, edge_type):
        """
        Validating the input of a node-searching method.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

        Returns
        -------
        None

        See Also
        --------
        `get_neighbors()`
        `get_parents()`
        `get_children()`
        `get_ancestors()`
        `get_descendants()`
        `get_spouses()`
        `get_reachable_nodes()`

        """
        # Check node's value
        if node is None:
            raise ValueError("Node cannot be None.")
        if node not in self.nodes():
            raise ValueError(f"Node {node} not in graph.")

        # Check edge_type's value
        if (edge_type is not None) and (edge_type not in self.SUPPORTED_EDGE_TYPES):
            raise ValueError(f"Types must be one of {self.SUPPORTED_EDGE_TYPES}.")

    def _preprocess_edge(
        self,
        u: Hashable,
        v: Hashable,
        edge_type: str,
    ) -> tuple[Hashable, Hashable, Hashable]:
        if edge_type == "<-":
            return (v, u, "->")
        elif edge_type == "o-":
            return (v, u, "-o")
        elif edge_type == "<o":
            return (v, u, "o>")
        elif edge_type == "<>":
            return (u, v, ">>")
        else:
            return (u, v, edge_type)

    def _unpreprocess_edge(
        self,
        u: Hashable,
        v: Hashable,
        edge_type: dict,
    ) -> tuple[Hashable, Hashable, Hashable]:
        u_symbol = edge_type[u]
        v_symbol = edge_type[v]
        edge_type = f"{u_symbol}{v_symbol}"

        if edge_type == ">>":
            edge_type = "<>"

        return u, v, edge_type
