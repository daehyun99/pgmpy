from typing import Hashable, Iterable, Optional, Union

import networkx as nx

from pgmpy.base._mixin_roles import _GraphRolesMixin


class _CoreGraph(nx.MultiGraph, _GraphRolesMixin):
    """
    Base graph class.

    Parameters
    ----------
    ebunch : input graph (optional, default: `None`)
    latents : set of nodes (default: empty `set()`)
    exposures : set of nodes (default: empty `set()`)
    outcomes : set of nodes (default: empty `set()`)
    roles : dict, optional (default: `None`)

    Examples
    --------
    Create an empty _CoreGraph with no nodes and no edges.

    >>> from pgmpy.base import _CoreGraph
    >>> G = _CoreGraph()

    Edges and vertices can be passed to the constructor as an edge list.

    >>> [Example_Code]

    **Nodes:**

    Add one node,

    >>> [Example_Code]

    Add a list of nodes,

    >>> [Example_Code]

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> [Example_Code]

    Add a list of edges,

    >>> [Example_Code]

    Remove one edge,

    >>> [Example_Code]

    Remove a list of edges,

    >>> [Example_Code]

    **Exposures, Outcomes, and Latents:**
        [Explain] about difference between Roles and Exposures, Outcomes, and Latents.

    >>> [Example_Code]

    **Roles:**

    >>> [Example_Code]

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

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G = _CoreGraph()

        """
        self._validating_edges_value(ebunch=[(u, v, type, key)])
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

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G = _CoreGraph()

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
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G = _CoreGraph()

        """
        self._validating_edges_value(ebunch=[(u, v, type, key)])

        if self.has_edge(u, v) == False:
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
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G = _CoreGraph()

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
        ebunch = []
        edges_keys = self.edges  # list(tuple(u, v, key), tuple(u, v, key), ...)
        edges_types = self.edges(  # list(tuple(u, v, type), tuple(u, v, type), ...)
            data=True
        )

        for (u, v, key), (_, _, type) in zip(edges_keys, edges_types):
            ebunch.append((u, v, type.get("type"), key))

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

        return (
            set(self.nodes()) == set(other.nodes())
            and set(self.edges()) == set(other.edges())
            and list(self.edges(data=True)) == list(other.edges(data=True))
            and self.exposures == other.exposures
            and self.outcomes == other.outcomes
            and self.latents == other.latents
            and self.get_role_dict() == other.get_role_dict()
        )

    def _validating_edges_value(
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
            `add_edge()`,
            `remove_edge()`.

        Returns
        -------
        None
        """
        for edge_type_key in ebunch:
            if len(edge_type_key) == 4:
                u, v, type, _ = edge_type_key
            else:
                u, v, type = edge_type_key
            if (u is None) or (v is None):
                raise ValueError("Nodes cannot be None.")
            if u == v:
                raise ValueError("Nodes cannot be the same for an edge.")
            if (type is None) or (type not in self.SUPPORTED_EDGE_TYPES):
                raise ValueError(f"Types must be one of {self.SUPPORTED_EDGE_TYPES}.")

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
