import inspect
from abc import ABC, abstractmethod
from typing import Hashable, Iterable, Optional

import networkx as nx

from pgmpy.base._mixin_roles import _GraphRolesMixin


class _CoreGraphABC(ABC):
    """
    An abstract class.

    Inheriting from `CoreGraph` automatically includes inheritance from `_CoreGraphABC`.
    """

    # Constants that the graph class must implement.
    ALLOWED_EDGES = None
    IS_DIRECTED = None
    IS_MULTIGRAPH = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if inspect.isabstract(cls):
            return

        if cls.IS_DIRECTED is None:
            raise TypeError("IS_DIRECTED must be defined.")

        if cls.IS_MULTIGRAPH is None:
            raise TypeError("IS_MULTIGRAPH must be defined.")

        if cls.ALLOWED_EDGES is None:
            raise TypeError("ALLOWED_EDGES must be defined.")

    @abstractmethod
    def _graph_type_marker(self):
        """
        This is a pure virtual method to prevent this class (and CoreGraph) from being instantiated directly.
        Concrete classes like DAG, PDAG, etc., must implement it.
        """
        pass


class CoreGraph(_CoreGraphABC, nx.MultiGraph, _GraphRolesMixin):
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
    Create an empty CoreGraph with no nodes and no edges.

    >>> from pgmpy.base import CoreGraph
    >>> G = CoreGraph()

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

    def is_directed(self):
        """Returns True if graph is directed, False otherwise."""
        return self.IS_DIRECTED

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return self.IS_MULTIGRAPH

    def add_edge(
        self,
        u: Hashable,  # Source node
        v: Hashable,  # Destination node
        type: tuple,  # (type at u, type at v) - type can be one of `-`, `>`, `o`
        **kwargs,
    ):
        """
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import CoreGraph
        >>> G = CoreGraph()

        """
        super().add_edge(u, v)

    def add_edges_from(self):
        """
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import CoreGraph
        >>> G = CoreGraph()

        """
        ...

    def remove_edge(self):
        """
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import CoreGraph
        >>> G = CoreGraph()

        """
        ...

    def remove_edges_from(self):
        """
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import CoreGraph
        >>> G = CoreGraph()

        """
        ...

    def __eq__(self):
        """
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import CoreGraph
        >>> G = CoreGraph()

        """
        ...

    def copy(self):
        """
        [Explain].

        Parameters
        ----------

        Examples
        --------
        >>> from pgmpy.base import CoreGraph
        >>> G = CoreGraph()

        """
        ...
