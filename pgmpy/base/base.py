from collections import deque
from typing import Hashable, Iterable, Optional

import networkx as nx

from pgmpy.base._mixin_roles import _GraphRolesMixin


class _CoreGraph(nx.MultiDiGraph, _GraphRolesMixin):
    """
    Base graph class for pgmpy.

    This class provides a generalized representation for all graph types in pgmpy.
    All specific graph classes (e.g., DAG, PDAG, ...) inherit from _CoreGraph.

    Subclasses are expected to define their own `SUPPORTED_EDGE_TYPES` to restrict the kinds of edges they can store.

    It also provides generalized algorithms and methods that work across all inheriting graph types.

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

    >>> from pgmpy.base import _CoreGraph
    >>> G = _CoreGraph()

    Edges and vertices can be passed to the constructor as an edge list.

    >>> edges = [("A", "B", "->"), ("B", "C", "->")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> G.get_edges(type=True)
    [('A', 'B', {'type': '->'}), ('B', 'C', {'type': '->'})]

    **Nodes:**

    Add one node,

    >>> from pgmpy.base import _CoreGraph
    >>> G = _CoreGraph()
    >>> G.add_node("A")
    >>> G.nodes
    NodeView(('A',))

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> from pgmpy.base import _CoreGraph
    >>> G = _CoreGraph()
    >>> G.add_edge("A", "B", "->")
    >>> G.get_edges(type=True)
    [('A', 'B', {'type': '->'})]

    Remove one edge,

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> G.remove_edge("A", "B", "->")
    >>> G.get_edges(type=True)
    [('B', 'C', {'type': '->'}), ('C', 'D', {'type': '--'})]

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
    >>> G.with_role("Custom_role", "A", inplace=True)
    >>> G.with_role("latents", "D", inplace=True)
    >>> G.get_role_dict()
    {'latents': ['D'], 'Custom_role': ['A']}
    >>> G.without_role("Custom_role", "A", inplace=True)
    >>> G.get_role_dict()
    {'latents': ['D']}
    """

    SUPPORTED_EDGE_TYPES = set(["--", "-o", "o-", "->", "<-", "o>", "<o", "<>", "oo"])

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
        type: str = None,
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

        type : str
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        None

        See Also
        --------
        `add_edges_from()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.
        The type value is matched with the key value of networkx.

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G = _CoreGraph()
        >>> G.add_edge("A", "B", "->")
        >>> G.get_edges(type=True)
        [('A', 'B', {'type': '->'})]
        """
        self._validating_edges_value(ebunch=[(u, v, type)])

        # Adding edge base on type value.
        if type in ["->", "<-", "o>", "<o"]:
            if type in ["<-", "<o"]:
                u, v = v, u
                type = f"{type[1]}>"
            super().add_edge(u, v, key=type, **kwargs)
        elif type in ["--", "-o", "o-"]:
            reverse_type = f"{type[1]}{type[0]}"
            super().add_edge(u, v, key=type, **kwargs)
            super().add_edge(v, u, key=reverse_type, **kwargs)
        elif type in ["<>", "oo"]:
            super().add_edge(u, v, key=type, **kwargs)
            super().add_edge(v, u, key=type, **kwargs)
        else:
            raise AssertionError(
                "This should never happen."
                "If you see this error, please file an issue on the pgmpy GitHub."
            )

    def add_edges_from(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable, Hashable]],
        **kwargs,
    ):
        """
        Add all the edges in ebunch.

        Parameters
        ----------
        ebunch : list of tuples
            [(`u`, `v`, `type`), (`u`, `v`, `type`), ...]

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
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        >>> G.add_edges_from(ebunch=edges)
        >>> G.get_edges(type=True)
        [('A', 'B', {'type': '->'}), ('B', 'C', {'type': '->'})]
        """
        self._validating_edges_value(ebunch=ebunch)

        for u, v, type in ebunch:
            self.add_edge(u, v, type, **kwargs)

    def remove_edge(
        self,
        u: Hashable,
        v: Hashable,
        type: str = None,
    ):
        """
        Remove an edge between u and v.

        Parameters
        ----------
        u, v : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        type : str
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
        The type value is matched with the key value of networkx.

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> G.remove_edge("A", "B", "->")
        >>> G.get_edges(type=True)
        [('B', 'C', {'type': '->'}), ('C', 'D', {'type': '--'})]
        """
        self._validating_edges_value(ebunch=[(u, v, type)])

        # Removing edge base on `type` value.
        if type in ["->", "<-", "o>", "<o"]:
            if type in ["<-", "<o"]:
                u, v = v, u
                type = f"{type[1]}>"
            super().remove_edge(u, v, key=type)
        elif type in ["--", "-o", "o-"]:
            reverse_type = f"{type[1]}{type[0]}"
            super().remove_edge(u, v, key=type)
            super().remove_edge(v, u, key=reverse_type)
        elif type in ["<>", "oo"]:
            super().remove_edge(u, v, key=type)
            super().remove_edge(v, u, key=type)
        else:
            raise AssertionError(
                "This should never happen."
                "If you see this error, please file an issue on the pgmpy GitHub."
            )

    def remove_edges_from(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable, Hashable]],
    ):
        """
        Remove all the edges in ebunch.

        Parameters
        ----------
        ebunch : list of tuples
            [(`u`, `v`, `type`), (`u`, `v`, `type`), ...]

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
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> remove_edges = [("B", "C", "->"), ("C", "D", "--")]
        >>> G.remove_edges_from(ebunch=remove_edges)
        >>> G.get_edges(type=True)
        [('A', 'B', {'type': '->'})]
        """
        self._validating_edges_value(ebunch=ebunch)

        for u, v, type in ebunch:
            self.remove_edge(u, v, type)

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


        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G1 = _CoreGraph()
        >>> G2 = G1.copy()
        >>> G2.__class__
        pgmpy.base.base._CoreGraph
        """
        ebunch = self._get_edges_type_key()

        graph_copy = self.__class__()
        graph_copy.add_edges_from(ebunch=ebunch)
        graph_copy.exposures = self.exposures
        graph_copy.outcomes = self.outcomes
        graph_copy.latents = self.latents
        for role, vars in self.get_role_dict().items():
            graph_copy.with_role(role=role, variables=vars, inplace=True)

        return graph_copy

    def get_edges(self, type: bool = False):
        """
        Returns a list of edges in the graph.

        For undirected and bidirected edges, which are stored as two directed
        edges internally, this method returns only one of them.

        Parameters
        ----------
        type: bool (default: False)
            If True, returns edge data. The edge data is a dict with 'type'.

        Returns
        -------
        list of tuples:
            If `type` is `True`, return `(u, v, {'type': type})`,
            and if `False`, return `(u, v)`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G = _CoreGraph(ebunch=[("A", "B", "->"), ("B", "C", "--")])
        >>> G.get_edges()
        [('A', 'B'), ('B', 'C')]
        >>> G.get_edges(type=True)
        [('A', 'B', {'type': '->'}), ('B', 'C', {'type': '--'})]
        """
        edges_type = self._get_edges_type_key()
        result = []
        seen = set()

        # Removing duplicates
        for u, v, type_val in edges_type:
            if (u, v, type_val) not in seen:
                seen.add((u, v, type_val))
                if type_val in ["--", "<>", "oo"]:
                    seen.add((v, u, type_val))
                elif type_val in ["-o", "o-"]:
                    reverse_type = f"{type_val[1]}{type_val[0]}"
                    seen.add((v, u, reverse_type))

                # Output format according to `type`
                if type:
                    output_form = (u, v, {"type": type_val})  # (u, v, data)
                else:
                    output_form = (u, v)  # (u, v)
                result.append(output_form)
        return result

    def get_neighbors(self, node, type=None):
        """
        Returns a set of neighbors nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        type : str
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
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        [Explain]
        """
        self._validating_nodes_value(node=node, type=type)

        if type is None:
            return set(nx.all_neighbors(self, node))

        result = set()
        if type in ["->", "o>", "--", "<>", "oo"]:
            for neighbor in nx.all_neighbors(self, node):
                if self.has_edge(node, neighbor, key=type):
                    result.add(neighbor)
        elif type in ["<-", "<o"]:
            reverse_type = f"{type[1]}>"
            for neighbor in nx.all_neighbors(self, node):
                if self.has_edge(neighbor, node, key=reverse_type):
                    result.add(neighbor)
        elif type in ["-o", "o-"]:
            reverse_type = f"{type[1]}{type[0]}"
            for neighbor in nx.all_neighbors(self, node):
                if (self.has_edge(node, neighbor, key=type)) or (
                    self.has_edge(neighbor, node, key=reverse_type)
                ):
                    result.add(neighbor)
        else:
            raise AssertionError(
                "This should never happen."
                "If you see this error, please file an issue on the pgmpy GitHub."
            )
        return result

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
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        [Explain]
        """
        self._validating_nodes_value(node=node, type="<-")
        return self.get_neighbors(node=node, type="<-")

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
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        [Explain]
        """
        self._validating_nodes_value(node=node, type="->")
        return self.get_neighbors(node=node, type="->")

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
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        [Explain]
        """
        self._validating_nodes_value(node=node, type="<>")
        return self.get_neighbors(node=node, type="<>")

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
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        [Explain]
        """
        ancestors = set()
        visited = set()
        queue = deque(node)
        nodes = self.nodes()

        while queue:
            current = queue.popleft()
            if current not in nodes:
                raise ValueError(f"Node {current} not in graph.")
            elif current not in visited:
                visited.add(current)
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
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        [Explain]
        """
        descendants = set()
        visited = set()
        queue = deque(node)
        nodes = self.nodes()

        while queue:
            current = queue.popleft()
            if current not in nodes:
                raise ValueError(f"Node {current} not in graph.")
            if current not in visited:
                visited.add(current)
                descendants.add(current)
                queue.extend(self.get_children(current))
        return descendants

    def get_reachable_nodes(self, node, type):
        """
        Returns a set of reachable nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        type : str
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
        >>> from pgmpy.base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        [Explain]
        """
        reachable = set()
        visited = set()
        queue = deque(node)
        nodes = self.nodes()

        while queue:
            current = queue.popleft()
            if current not in nodes:
                raise ValueError(f"Node {current} not in graph.")
            if current not in visited:
                visited.add(current)
                reachable.add(current)
                queue.extend(self.get_neighbors(current, type=type))
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
        >>> from pgmpy.base import _CoreGraph
        >>> G1 = _CoreGraph()
        >>> G2 = _CoreGraph()
        >>> G1.__eq__(G2)
        True
        """
        if not isinstance(other, self.__class__):
            return False

        self_ebunch = self._get_edges_type_key()
        other_ebunch = other._get_edges_type_key()

        return (
            set(self.nodes()) == set(other.nodes())
            and set(self_ebunch) == set(other_ebunch)
            and self.exposures == other.exposures
            and self.outcomes == other.outcomes
            and self.latents == other.latents
            and self.get_role_dict() == other.get_role_dict()
        )

    def _validating_edges_value(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable, Hashable]],
    ):
        """
        Validates the value input by the user, then either raises an error.

        Parameters
        ----------
        ebunch : list of tuples
            [(`u`, `v`, `type`), (`u`, `v`, `type`), ...]

        Notes
        -----
        Helper method that validates the input for
            `add_edge()`,
            `add_edges_from()`,
            `remove_edge()`,
            `remove_edges_from()`.
        """
        for u, v, type in ebunch:
            if (u is None) or (v is None):
                raise ValueError("Nodes cannot be None.")
            if u == v:
                raise ValueError("Nodes cannot be the same for an edge.")
            if (type is None) or (type not in self.SUPPORTED_EDGE_TYPES):
                raise ValueError(f"Types must be one of {self.SUPPORTED_EDGE_TYPES}.")

    def _get_edges_type_key(self):
        """
        Returns the all edge's type value connecting the two nodes as a list of tuples.

        Parameters
        ----------
        None

        Returns
        -------
        ebunch : list of tuples
            [(`u`, `v`, `type`), (`u`, `v`, `type`), ...]

        Notes
        -----
        The type value is matched with the key value of networkx.
        Helper method for
            `copy()`,
            `__eq__()`,
            `get_edges()`,
        """
        return sorted(
            [(u, v, type) for u, v, type in self.edges(keys=True)],
            key=lambda x: (x[0], x[1]),
        )

    def _validating_nodes_value(self, node, type):
        """
        Validating the input of a node-searching method.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        type : str
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

        # Check type's value
        if (type is not None) and (type not in self.SUPPORTED_EDGE_TYPES):
            raise ValueError(f"Types must be one of {self.SUPPORTED_EDGE_TYPES}.")
