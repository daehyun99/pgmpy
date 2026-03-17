#!/usr/bin/env python3

from typing import Hashable, Set

import networkx as nx


class _GraphAlgorithmMixin:
    """Mixin class for causal graph's algorithms."""

    # ----------------------------------------------------------------------
    # Public API (or Public Methods)
    # ----------------------------------------------------------------------

    def is_collider(self, u: Hashable, v: Hashable, w: Hashable):
        """


        Parameters
        ----------
        u, v, w : src, dst, mid

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
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        if not {u, v, w}.issubset(self.nodes):
            raise ValueError(f"{u}, {v}, {w} must be present in the graph.")

        parents = self.get_parents(w)
        spouses = self.get_spouses(w)

        incoming_to_w = parents.union(spouses)

        return (u in incoming_to_w) and (v in incoming_to_w)

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
        # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        raise NotImplementedError("`is_m_separator` is not supported now")

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
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
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
        # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        raise NotImplementedError("`is_minimal_m_separator` is not supported now")

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
        # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        raise NotImplementedError("`get_m_separator` is not supported now")

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
        # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        raise NotImplementedError("`get_minimal_m_separator` is not supported now")

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
        # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        raise NotImplementedError("`get_m_separators` is not supported now")

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
        # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        raise NotImplementedError("`get_minimal_m_separators` is not supported now")

    def get_ancestral_graph(self, nodes):
        """
        Return the ancestral graph induced by the input nodes.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes to induce subgraph on.

        Returns
        -------
        Graph
            Subgraph induced by ancestors of the given nodes.

        Raises
        ------

        """
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)

        ancestors = set(nodes_set)
        for node in nodes_set:
            ancestor = self.get_ancestors(node)
            ancestors.update(ancestor)

        new_graph = type(self)()

        for u, v, key, data in self.get_edges(keys=True, data=True):
            if (u in ancestors) and (v in ancestors):
                new_graph.add_edge(u, v, edge_type=data, key=key)

        new_graph.add_nodes_from(ancestors)

        return new_graph

    def get_markov_blanket(self, nodes):
        """


        Parameters
        ----------

        Returns
        -------
        bool

        See Also
        --------
        `DAG`, `ADMG`

        Notes
        -----
        Currently, this is only applicable to ADMGs and DAGs.

        Examples
        --------

        References
        ----------
        [1] Richardson, Thomas. "Markov Properties for Acyclic Directed Mixed Graphs."
            Scandinavian Journal of Statistics 30.1 (2003): 145-157.
            https://doi.org/10.1111/1467-9469.00323
        """
        # NOTE: For simplicity of definition, current support is limited to DAGs and ADMGs.
        #       This can be extended to MAGs and PAGs in the future.
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)

        if not nodes_set.issubset(self.nodes):
            raise ValueError("Input nodes must be subset of graph's nodes.")

        markov_blanket = set()
        for node in nodes_set:
            markov_blanket.update(
                self.get_parents(node), self.get_children(node), self.get_spouses(node)
            )

        markov_blanket -= nodes_set

        return markov_blanket

    def has_inducing_path(self, u, v, W):
        """
        Need to modify

        Check if there exists an inducing path between two nodes relative to W.

        An inducing path between u and v is a path such that:
        - The path has length > 2 (at least one intermediate node),
        - Every intermediate node is a collider on the path,
        - Every intermediate node is either:
            * in W, or
            * an ancestor of u or v.

        Parameters
        ----------
        u : Hashable
            Source node.

        v : Hashable
            Target node.

        W : set
            Subset of nodes to check inducing paths through (often latents).

        Returns
        -------
        bool
            True if there exists an inducing path, False otherwise.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "L", "-", ">")
        >>> mag.add_edge("Y", "L", "-", ">")
        >>> mag.latents = {"L"}
        >>> mag.has_inducing_path("X", "Y", mag.latents)
        True
        """
        # TODO(@daehyun99): [#2385] Implement code logic and test code
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)

        has_inducing = False

        ancestors = self.get_ancestors(u)
        ancestors.update(self.get_ancestors(v))

        for path in nx.all_simple_paths(self, source=u, target=v):

            if len(path) <= 2:
                continue

            for i in range(len(path) - 3):
                src, mid, dst = path[i : i + 3]

                if self.is_collider(src, mid, dst) and mid in W:
                    has_inducing = True
                    break

                elif not self.is_collider(src, mid, dst) and min not in W:
                    has_inducing = True
                    break

                if mid in ancestors:
                    has_inducing = True
                    break

        return has_inducing

    def has_direct_path(self, u, v):
        """

        Parameters
        ----------

        Returns
        -------
        bool

        See Also
        --------
        `nx.has_path`

        Notes
        -----

        Examples
        --------

        References
        ----------

        """
        # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        for path in nx.all_simple_edge_paths(self, u, v):
            is_directed_path = True
            for edge in path:
                src, dst, key = edge
                if self[src][dst][key] != {src: "-", dst: ">"}:
                    is_directed_path = False
                    break
            if is_directed_path:
                return True
        return False

    def has_directed_cycle(self):
        """


        Parameters
        ----------

        Returns
        -------
        bool

        See Also
        --------
        `DAG`, `ADMG`

        Notes
        -----


        Examples
        --------

        References
        ----------
        [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
        Journal of Machine Learning Research 9 (2008): 1437-1474.
        """
        # TODO(@daehyun99): [#2385] Consider implement `has_directed_cycle` method.
        # Cycles can be easily identified using the has_direct_path method (refer to ADMG.add_edge()).
        # We will reassess the necessity of developing this method at a later date.
        # # TODO(@daehyun99): [#2385] Implement code logic and test code
        # # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        raise NotImplementedError("`has_directed_cycle` is not supported now")

    def has_almost_directed_cycle(self):
        """


        Parameters
        ----------

        Returns
        -------
        bool

        See Also
        --------
        `MAG`, `ADMG`

        Notes
        -----


        Examples
        --------

        References
        ----------
        [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
        Journal of Machine Learning Research 9 (2008): 1437-1474.
        """
        # TODO(@daehyun99): [#2385] Consider implement `has_almost_directed_cycle` method.
        # Cycles can be easily identified using the has_direct_path method (refer to ADMG.add_edge()).
        # We will reassess the necessity of developing this method at a later date.
        # # TODO(@daehyun99): [#2385] Implement code logic and test code
        # # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        raise NotImplementedError("`has_almost_directed_cycle` is not supported now")

    # ----------------------------------------------------------------------
    # Internal Methods (or Private Methods)
    # ----------------------------------------------------------------------
