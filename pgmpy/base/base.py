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
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
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
        self._validate_edges_value(ebunch=[(u, v, type, key)])
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
        self._validate_edges_value(ebunch=ebunch)

        for edge_type_key in ebunch:
            if len(edge_type_key) == 4:
                u, v, type, key = edge_type_key
            else:
                u, v, type = edge_type_key
                key = None
            self.add_edge(u, v, type, key=key, **kwargs)

    def remove_edge(self):
        """
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G = _CoreGraph()

        """
        ...

    def remove_edges_from(self):
        """
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G = _CoreGraph()

        """
        ...

    def copy(self):
        """
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G = _CoreGraph()

        """
        ...

    # ----------------------------------------------------------------------
    # Internal Methods (or Private Methods)
    # ----------------------------------------------------------------------

    def __eq__(self):
        """
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import _CoreGraph
        >>> G = _CoreGraph()

        """
        ...

    def _validate_edges_value(
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
            `add_edges_from()`,
            `remove_edge()`,
            and `remove_edges_from()`.
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
