from typing import Hashable, Iterable, Optional

import networkx as nx

from pgmpy.base._mixin_roles import _GraphRolesMixin


class _CoreGraph(nx.MultiGraph, _GraphRolesMixin):
    """
    Base graph class.

    Parameters
    ----------
    ebunch : input graph (optional, default: None)
    latents : set of nodes (default: empty set)
    exposures : set of nodes (default: empty set)
    outcomes : set of nodes (default: empty set)
    roles : dict, optional (default: None)

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

    SUPPORTED_EDGE_TYPES = ["--", "-*", "*-", "->", "<-", "*>", "<*", "<>", "**"]

    def __init__(
        self,
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
        exposures: set[Hashable] = set(),
        outcomes: set[Hashable] = set(),
        latents: set[Hashable] = set(),
        roles=None,
    ):
        super().__init__(ebunch)

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

    def add_edge(
        self,
        u: Hashable,
        v: Hashable,
        type: str,
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
        super().add_edge(u, v)

    def add_edges_from(self):
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
