#!/usr/bin/env python3

from typing import Hashable, Set


class _GraphAlgorithmMixin:
    """Mixin class for causal graph's algorithms."""

    def is_collider(self, u: Hashable, v: Hashable, w: Hashable):
        """


        Parameters
        ----------
        u, v, w : node

        Returns
        -------
        bool

        See Also
        --------

        Notes
        -----

        Examples
        --------

        References
        ----------
        [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
        Journal of Machine Learning Research 9 (2008): 1437-1474.
        """
        ...

    def is_m_separator(self, x: Set, y: Set, z: Set):
        """


        Parameters
        ----------

        Returns
        -------
        bool

        See Also
        --------
        `is_m_connected()`
        `is_minimal_m_separator()`

        Notes
        -----
        This implementation is based on the 'TestSep' algorithm [1].
        The pseudo-code logic is as follows:

        .. code-block:: text

            function TestSep(G, X, Y, Z)
                P <- { (Wait_Direction, x) | x in X }   # Pending visits
                Q <- P                                  # History (visited)

                while P is not empty do
                    Let (e, T) be a pair in P
                    Remove (e, T) from P

                    for all neighbors N of T do
                        Let T and N be connected by edge f
                        if (e, T, f) is m-connecting given Z and (f, N) not in Q then
                            Add (f, N) to P and Q
                return true

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...

    def is_m_connected(self, x: Set, y: Set, z: Set):
        """


        Parameters
        ----------

        Returns
        -------
        bool

        See Also
        --------
        `is_m_separator()`

        Notes
        -----

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        return not self.is_m_separator(x, y, z)

    def is_minimal_m_separator(self, x: Set, y: Set, z: Set):
        """


        Parameters
        ----------

        Returns
        -------
        bool

        See Also
        --------
        `is_m_separator()`

        Notes
        -----
        This implementation is based on the 'TestMinSep' algorithm [1].
        The pseudo-code logic is as follows:

        .. code-block:: text
            function TestMinSep(G, X, Y, Z, M, R)
                if (Z - Ant(X U Y U M)) is not empty or Z is not subset of R then return false
                if not TestSep(G, X, Y, Z) then return false
                G'a <- (G_Ant(X U Y U M))^a
                Remove from G'a all nodes of M
                Rx <- { z in Z | exists a path from X to z in G'a not intersecting Z - {z} }
                if Z != Rx then return false
                Ry <- { z in Z | exists a path from Y to z in G'a not intersecting Z - {z} }
                if Z != Ry then return false
                return true

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...

    def get_m_separator(self, x: Set, y: Set, i: Set, r: Set):
        """


        Parameters
        ----------

        Returns
        -------
        M-Separation: set


        See Also
        --------
        `get_minimal_m_separator()`

        Notes
        -----
        This implementation is based on the 'FindSep' algorithm [1].
        The pseudo-code logic is as follows:

        .. code-block:: text

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...

    def get_minimal_m_separator(self, x: Set, y: Set, i: Set, r: Set):
        """


        Parameters
        ----------

        Returns
        -------
        M-Separation: set


        See Also
        --------
        `get_m_separator()`

        Notes
        -----
        This implementation is based on the 'FindMinSep' algorithm [1].
        The pseudo-code logic is as follows:

        .. code-block:: text
            function FindMinSep(G, X, Y, I, R)
                G' <- G_Ant(X U Y U I)                       // Ancestral subgraph containing X, Y, I
                G'a <- (G_Ant(X U Y U I))^a                  // Moral graph of the ancestral subgraph
                Remove from G'a all nodes of I
                Z' <- R intersection Ant(X U Y) - (X U Y)    // Candidates restricted to R and ancestors
                Z'' <- { Z in Z' | exists a path from X to Z in G'a not intersecting Z' - {Z} }
                Z <- { Z in Z'' | exists a path from Y to Z in G'a not intersecting Z'' - {Z} }
                if not TestSep(G', X, Y, Z) then return ⊥
                return Z U I

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...

    def get_m_separators(self, x: Set, y: Set, i: Set, r: Set):
        """


        Parameters
        ----------

        Returns
        -------
        List of M-Separation: list of set

        See Also
        --------
        `get_minimal_m_separators()`

        Notes
        -----
        This implementation is based on the 'ListSep' algorithm [1].
        The pseudo-code logic is as follows:

        .. code-block:: text
            function ListSep(G, X, Y, I, R)
                if FindSep(G, X, Y, I, R) != ⊥ then
                    if I = R then
                        Output I
                    else
                        V <- an arbitrary node of R - I
                        ListSep(G, X, Y, I U {V}, R)
                        ListSep(G, X, Y, I, R - {V})

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...

    def get_minimal_m_separators(self, x: Set, y: Set, i: Set, r: Set):
        """


        Parameters
        ----------

        Returns
        -------
        List of M-Separation: list of set

        See Also
        --------
        `get_m_separators()`

        Notes
        -----
        This implementation is based on the 'ListMinSep' algorithm [1].
        The pseudo-code logic is as follows:

        .. code-block:: text
            function ListMinSep(G, X, Y, I, R)
                G' <- G_Ant(X U Y U I)
                G'a <- (G_Ant(X U Y U I))^a
                Add a node X_m connected to all X nodes
                Add a node Y_m connected to all Y nodes
                Remove nodes of I
                Remove nodes of V - R connecting the neighbors of each removed node
                Use the algorithm in [2] to list all sets separating X_m and Y_m
        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.

        [2] Takata, Ken.
        "Space-optimal, backtracking algorithms to list the minimal vertex separators of a graph."
        Discrete Applied Mathematics 158 (2010): 1660-1667.
        """
        ...
