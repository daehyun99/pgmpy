from abc import ABC, abstractmethod
from typing import Hashable, Iterable, Optional

import networkx as nx

from pgmpy.base._mixin_roles import _GraphRolesMixin


class CoreGraphABC(ABC):
    """
    An abstract class that all graphs must implement.
    It must be the first class inherited by all graphs.
    """

    @abstractmethod
    def is_directed(self):
        """Returns True if graph is directed, False otherwise."""
        pass

    @abstractmethod
    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        pass


class CoreGraph(CoreGraphABC, nx.MultiGraph, _GraphRolesMixin):
    """
    Base graph class.

    Parameters
    ----------

    Examples
    --------
    >>> from pgmpy.base import CoreGraph
    >>> G = CoreGraph()

    """

    def __init__(
        self,
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
        latents: set[Hashable] = set(),
        roles=None,
    ):
        super().__init__(ebunch)

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
        return False

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return True

    def add_edge(self):
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
