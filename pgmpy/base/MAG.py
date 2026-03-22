from typing import Hashable, Iterable, Optional

from pgmpy.base._base import _CoreGraph


class MAG(_CoreGraph):
    """
    Class for representing Maximal Ancestral Graphs (MAGs).

    A MAG is a type of graph used in causal inference to represent conditional
    independence relations when some variables are latent (unobserved). Unlike
    simple directed acyclic graphs (DAGs), MAGs allow for special edge types
    (directed and bidirected) that capture the presence of latent confounding
    and selection bias. Every pair of nodes in a MAG is connected in such a way
    that the graph is "maximal," meaning no additional edges can be added
    without changing the set of implied conditional independence relations.


    Parameters
    ----------
    ebunch : iterable of tuples, optional
        A list or iterable of edges to add at initialization.

    latents : set, default=set()
        Set of latent (unobserved) variables.

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
        If provided, this will automatically assign roles to the nodes in the graph.
        Passing a key-value pair via ``roles`` is equivalent to calling
        ``with_role(role, variables)`` for each key-value pair in the dictionary.

    Examples
    --------
    >>> from pgmpy.base import MAG
    >>> mag = MAG(ebunch=[("L", "A", "-", ">"), ("B", "C", "-", ">")], latents={"L"})
    >>> sorted(mag.nodes())
    ['A', 'B', 'C', 'L']

    Roles can be assigned to nodes in the graph at construction or using methods.

    At construction:

    >>> mag = MAG(
    ...     ebunch=[("L", "A", "-", ">"), ("B", "C", "-", ">")],
    ...     latents={"L"},
    ...     exposures={"A"},
    ...     outcomes={"B"},
    ... )

    Roles can also be assigned after creation using ``with_role`` method.

    >>> mag = mag.with_role("adjustment", {"L", "C"})

    Vertices of a specific role can be retrieved using ``get_role`` method.

    >>> mag.get_role("exposures")
    ["A"]
    >>> mag.get_role("adjustment")
    ["L", "C"]

    References
    ----------
    .. [1] Zhang, J. (2008). Causal Reasoning with Ancestral Graphs. Journal of Machine Learning Research, 9(7).
    """

    SUPPORTED_EDGE_TYPES = frozenset(["--", "->", "<-", "<>"])

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable, Hashable]] = None,
        exposures: Optional[set[Hashable]] = None,
        outcomes: Optional[set[Hashable]] = None,
        latents: Optional[set[Hashable]] = None,
        roles=None,
    ):
        super().__init__(
            ebunch=ebunch,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

    def to_pag(self):
        """

        Parameters
        ----------

        Returns
        -------
        Graphs

        See Also
        --------
        `MAG`, `PAG`

        Notes
        -----

        Examples
        --------

        References
        ----------

        """
        # TODO: Implement method when Refactor PAG
        # TODO: Fix Docs (Unify Docs Format)
        # TODO: Apply type hint(input, output)
        raise NotImplementedError("`to_pag` is not supported now")

    def is_valid_mag(self):
        """
        checking is mag
        - the graph does not contain any directed or almost directed cycles (ancestral)
        - there is no inducing path between any two non-adjacent vertices (maximal)

        Parameters
        ----------

        Returns
        -------
        Graphs

        See Also
        --------
        `MAG`, `PAG`

        Notes
        -----

        Examples
        --------

        References
        ----------
        [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
        Journal of Machine Learning Research 9 (2008): 1437-1474.
        """
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)

        # # TODO(@daehyun99): [#2385] Implement code logic and test code When Refactor DAG
        # if self.has_directed_cycle() or self.has_almost_directed_cycle():
        # return False

        # TODO(@daehyun99): [#2385] Checking inducing path between any two non-adjacent vertices
        #                   has_inducing_path
        # return True
        raise NotImplementedError("`is_valid_mag` is not supported now")

    def _validate_graph_specific_edges(
        self,
        ebunch: (
            Iterable[tuple[Hashable, Hashable, Hashable]]
            | Iterable[tuple[Hashable, Hashable, Hashable, Hashable]]
        ),
    ):
        # TODO(@daehyun99): [#2385] Implement Checking MAG's rule.
        # self.is_mag()
        pass
