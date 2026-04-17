#!/usr/bin/env python3

from collections.abc import Hashable, Iterable

import networkx as nx

from pgmpy.utils.types import Self


class _GraphAlgorithmMixin:
    """Mixin class for causal graph's algorithms."""

    # ----------------------------------------------------------------------
    # Public API (or Public Methods)
    # ----------------------------------------------------------------------

    def is_collider(self, u: Hashable, v: Hashable, w: Hashable) -> bool:
        """
        Check whether `w` is a collider between `u` and `v`.

        Parameters
        ----------
        u : Hashable
            The first endpoint node.
        v : Hashable
            The second endpoint node.
        w : Hashable
            The middle node to test as a collider.

        Returns
        -------
        bool

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph()
        >>> graph.add_edge("T", "M", "->")
        >>> graph.add_edge("M", "O", "->")
        >>> graph.add_edge("M", "I", "<-")
        >>> graph.add_edge("M", "B", "<>")
        >>> graph.add_edge("M", "U", "--")
        >>> graph.is_collider("T", "O", "M")
        False
        >>> graph.is_collider("T", "I", "M")
        True
        >>> graph.is_collider("T", "B", "M")
        True
        >>> graph.is_collider("T", "U", "M")
        False

        References
        ----------
        [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
        Journal of Machine Learning Research 9 (2008): 1437-1474.
        """
        if not {u, v, w}.issubset(self.nodes):
            raise ValueError(f"{u}, {v}, {w} must be present in the graph.")

        neighbors = self.neighbors(u)
        if v in neighbors:
            return False

        parents = self.get_neighbors(w, edge_type="<-")
        spouses = self.get_neighbors(w, edge_type="<>")

        incoming_to_w = parents.union(spouses)

        return (u in incoming_to_w) and (v in incoming_to_w)

    # def is_m_separator(self, x: set, y: set, z: set):
    #     """

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     bool

    #     See Also
    #     --------
    #     `is_m_connected()`
    #     `is_minimal_m_separator()`

    #     Notes
    #     -----
    #     This implementation is based on the 'TestSep' algorithm [1].
    #     The pseudo-code logic is as follows:

    #     .. code-block:: text

    #         function TestSep(G, X, Y, Z)
    #             P <- { (Wait_Direction, x) | x in X }   # Pending visits
    #             Q <- P                                  # History (visited)

    #             while P is not empty do
    #                 Let (e, T) be a pair in P
    #                 Remove (e, T) from P

    #                 for all neighbors N of T do
    #                     Let T and N be connected by edge f
    #                     if (e, T, f) is m-connecting given Z and (f, N) not in Q then
    #                         Add (f, N) to P and Q
    #             return true

    #     Examples
    #     --------

    #     References
    #     ----------
    #     [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40.
    #     """
    #     # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
    #     # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
    #     # TODO(@daehyun99): [#2385] Apply type hint(input, output)
    #     raise NotImplementedError("`is_m_separator` is not supported now")

    # def is_m_connected(self, x: set, y: set, z: set):
    #     """

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     bool

    #     See Also
    #     --------
    #     `is_m_separator()`

    #     Notes
    #     -----

    #     Examples
    #     --------

    #     References
    #     ----------
    #     [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40.
    #     """
    #     # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
    #     # TODO(@daehyun99): [#2385] Apply type hint(input, output)
    #     return not self.is_m_separator(x, y, z)

    # def is_minimal_m_separator(self, x: set, y: set, z: set):
    #     """

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     bool

    #     See Also
    #     --------
    #     `is_m_separator()`

    #     Notes
    #     -----
    #     This implementation is based on the 'TestMinSep' algorithm [1].
    #     The pseudo-code logic is as follows:

    #     .. code-block:: text
    #         function TestMinSep(G, X, Y, Z, M, R)
    #             if (Z - Ant(X U Y U M)) is not empty or Z is not subset of R then return false
    #             if not TestSep(G, X, Y, Z) then return false
    #             G'a <- (G_Ant(X U Y U M))^a
    #             Remove from G'a all nodes of M
    #             Rx <- { z in Z | exists a path from X to z in G'a not intersecting Z - {z} }
    #             if Z != Rx then return false
    #             Ry <- { z in Z | exists a path from Y to z in G'a not intersecting Z - {z} }
    #             if Z != Ry then return false
    #             return true

    #     Examples
    #     --------

    #     References
    #     ----------
    #     [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40.
    #     """
    #     # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
    #     # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
    #     # TODO(@daehyun99): [#2385] Apply type hint(input, output)
    #     raise NotImplementedError("`is_minimal_m_separator` is not supported now")

    # def get_m_separator(self, x: set, y: set, i: set, r: set):
    #     """

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     M-Separation: set

    #     See Also
    #     --------
    #     `get_minimal_m_separator()`

    #     Notes
    #     -----
    #     This implementation is based on the 'FindSep' algorithm [1].
    #     The pseudo-code logic is as follows:

    #     .. code-block:: text

    #     Examples
    #     --------

    #     References
    #     ----------
    #     [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40.
    #     """
    #     # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
    #     # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
    #     # TODO(@daehyun99): [#2385] Apply type hint(input, output)
    #     raise NotImplementedError("`get_m_separator` is not supported now")

    # def get_minimal_m_separator(self, x: set, y: set, i: set, r: set):
    #     """

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     M-Separation: set

    #     See Also
    #     --------
    #     `get_m_separator()`

    #     Notes
    #     -----
    #     This implementation is based on the 'FindMinSep' algorithm [1].
    #     The pseudo-code logic is as follows:

    #     .. code-block:: text
    #         function FindMinSep(G, X, Y, I, R)
    #             G' <- G_Ant(X U Y U I)                       // Ancestral subgraph containing X, Y, I
    #             G'a <- (G_Ant(X U Y U I))^a                  // Moral graph of the ancestral subgraph
    #             Remove from G'a all nodes of I
    #             Z' <- R intersection Ant(X U Y) - (X U Y)    // Candidates restricted to R and ancestors
    #             Z'' <- { Z in Z' | exists a path from X to Z in G'a not intersecting Z' - {Z} }
    #             Z <- { Z in Z'' | exists a path from Y to Z in G'a not intersecting Z'' - {Z} }
    #             if not TestSep(G', X, Y, Z) then return ⊥
    #             return Z U I

    #     Examples
    #     --------

    #     References
    #     ----------
    #     [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40.
    #     """
    #     # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
    #     # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
    #     # TODO(@daehyun99): [#2385] Apply type hint(input, output)
    #     raise NotImplementedError("`get_minimal_m_separator` is not supported now")

    # def get_m_separators(self, x: set, y: set, i: set, r: set):
    #     """

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     List of M-Separation: list of set

    #     See Also
    #     --------
    #     `get_minimal_m_separators()`

    #     Notes
    #     -----
    #     This implementation is based on the 'ListSep' algorithm [1].
    #     The pseudo-code logic is as follows:

    #     .. code-block:: text
    #         function ListSep(G, X, Y, I, R)
    #             if FindSep(G, X, Y, I, R) != ⊥ then
    #                 if I = R then
    #                     Output I
    #                 else
    #                     V <- an arbitrary node of R - I
    #                     ListSep(G, X, Y, I U {V}, R)
    #                     ListSep(G, X, Y, I, R - {V})

    #     Examples
    #     --------

    #     References
    #     ----------
    #     [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40.
    #     """
    #     # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
    #     # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
    #     # TODO(@daehyun99): [#2385] Apply type hint(input, output)
    #     raise NotImplementedError("`get_m_separators` is not supported now")

    # def get_minimal_m_separators(self, x: set, y: set, i: set, r: set):
    #     """

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     List of M-Separation: list of set

    #     See Also
    #     --------
    #     `get_m_separators()`

    #     Notes
    #     -----
    #     This implementation is based on the 'ListMinSep' algorithm [1].
    #     The pseudo-code logic is as follows:

    #     .. code-block:: text
    #         function ListMinSep(G, X, Y, I, R)
    #             G' <- G_Ant(X U Y U I)
    #             G'a <- (G_Ant(X U Y U I))^a
    #             Add a node X_m connected to all X nodes
    #             Add a node Y_m connected to all Y nodes
    #             Remove nodes of I
    #             Remove nodes of V - R connecting the neighbors of each removed node
    #             Use the algorithm in [2] to list all sets separating X_m and Y_m
    #     Examples
    #     --------

    #     References
    #     ----------
    #     [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40.

    #     [2] Takata, Ken.
    #     "Space-optimal, backtracking algorithms to list the minimal vertex separators of a graph."
    #     Discrete Applied Mathematics 158 (2010): 1660-1667.
    #     """
    #     # TODO(@daehyun99): [#2385], [#2342] Implement `m-separation`
    #     # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
    #     # TODO(@daehyun99): [#2385] Apply type hint(input, output)
    #     raise NotImplementedError("`get_minimal_m_separators` is not supported now")

    def get_ancestral_graph(self, nodes: str | Iterable[str]) -> Self:
        """
        Return the ancestral graph induced by the input nodes.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes to induce subgraph on.

        Returns
        -------
        ancestral_graph: Self
            ancestral graph object

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [
        ...     ("A", "B", "->"),
        ...     ("B", "C", "->"),
        ...     ("C", "D", "<>"),
        ...     ("C", "E", "--"),
        ... ]
        >>> graph = _CoreGraph()
        >>> graph.add_edges_from(edges)
        >>> ancestral_graph = graph.get_ancestral_graph("C")
        >>> ancestral_graph.get_edges(keys=True, data=True)
        [("A", "B", 0, "->"), ("B", "C", 0, "->")]

        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)

        ancestors = set(nodes_set)
        for node in nodes_set:
            ancestor = self.get_ancestors(node)
            ancestors.update(ancestor)

        return self.subgraph(ancestors).copy()

    def get_markov_blanket(self, nodes: str | Iterable[str]) -> str | Iterable[str]:
        """
        Return the Markov blanket of the given node or nodes.

        Parameters
        ----------
        nodes : str or iterable of str
            A node name or an iterable of node names whose Markov blanket is to be
            computed.

        Returns
        -------
        markov_blanket: set
            A set containing the nodes in the Markov blanket of the input node or
            nodes.

        See Also
        --------
        `DAG`, `ADMG`

        Notes
        -----
        Currently, this method is only applicable to ADMGs and DAGs.

        Examples
        --------
        >>> edges = [
        ...     ("A", "B", "->"),
        ...     ("B", "C", "->"),
        ...     ("D", "E", "->"),
        ...     ("A", "D", "<>"),
        ...     ("B", "E", "<>"),
        ... ]
        >>> admg = ADMG()
        >>> admg.add_edges_from(edges)
        >>> admg.add_node("F", latent=True)
        >>> admg.with_role(role="exposures", variables={"A"}, inplace=True)
        >>> admg.with_role(role="outcomes", variables={"C"}, inplace=True)

        >>> admg.get_markov_blanket("B")
        {'A', 'C', 'E'}

        References
        ----------
        [1] Richardson, Thomas. "Markov Properties for Acyclic Directed Mixed Graphs."
            Scandinavian Journal of Statistics 30.1 (2003): 145-157.
            https://doi.org/10.1111/1467-9469.00323
        """
        # NOTE: For simplicity of definition, current support is limited to DAGs and ADMGs.
        #       This can be extended to MAGs and PAGs in the future.
        from pgmpy.base import ADMG, DAG

        if not (isinstance(self, DAG) or isinstance(self, ADMG)):
            raise TypeError("get_markov_blanket is currently only supported for ADMGs and DAGs.")

        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)

        if not nodes_set.issubset(self.nodes):
            raise ValueError("Input nodes must be subset of graph's nodes.")

        markov_blanket = set()
        for node in nodes_set:
            markov_blanket.update(self.get_parents(node), self.get_children(node), self.get_spouses(node))

        markov_blanket -= nodes_set

        return markov_blanket

    def has_inducing_path(self, u: Hashable, v: Hashable, w: set) -> bool:
        """
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

        w : set
            Subset of nodes to check inducing paths through (often latents).

        Returns
        -------
        bool
            True if there exists an inducing path, False otherwise.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [
        ...     ("A", "B", "<>"),
        ...     ("A", "C", "<>"),
        ...     ("B", "D", "<>"),
        ...     ("A", "D", "->"),
        ...     ("B", "C", "->"),
        ... ]
        >>> graph = _CoreGraph(ebunch=edges)
        >>> graph.has_inducing_path("C", "D", set())
        True

        """
        has_inducing = False

        ancestors = self.get_ancestors(u)
        ancestors.update(self.get_ancestors(v))

        for path in nx.all_simple_paths(self, source=u, target=v):
            if len(path) <= 2:
                continue

            for i in range(len(path) - 3):
                src, mid, dst = path[i : i + 3]

                if self.is_collider(src, mid, dst) and mid in w:
                    has_inducing = True
                    break

                elif not self.is_collider(src, mid, dst) and mid not in w:
                    has_inducing = True
                    break

                if mid in ancestors:
                    has_inducing = True
                    break

        return has_inducing

    def has_direct_path(self, u: Hashable, v: Hashable) -> bool:
        """
        Check whether there exists a fully directed path from `u` to `v`.

        Parameters
        ----------
        u : Hashable
            The source node.

        v : Hashable
            The destination node.

        Returns
        -------
        bool

        See Also
        --------
        `nx.has_path`

        Examples
        --------
        >>> graph = _CoreGraph()
        >>> graph.add_edge("A", "B", "->")
        >>> graph.add_edge("B", "C", "->")
        >>> graph.has_direct_path("A", "C")
        True

        """
        for path in nx.all_simple_edge_paths(self, u, v):
            is_directed_path = True
            for edge in path:
                src, dst, key = edge
                edge_type = self.get_edge_type(src, dst, key)
                if edge_type != "->":
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
        `DAG`
        `ADMG`
        `PDAG`
        `MAG`

        Notes
        -----


        Examples
        --------

        References
        ----------
        [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
        Journal of Machine Learning Research 9 (2008): 1437-1474.
        """
        # # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        # # TODO(@daehyun99): [#2385] Implement code logic and test code When Refactor DAG
        # networkx_ebunch = super().edges(keys=True, data=True)
        # from pgmpy.base import DAG
        # dag = DAG()
        # for edge in networkx_ebunch:
        # if edge[-1] == {edge[0]: "-", edge[1]: ">"}:
        # dag.add_edge(edge[0], edge[1], "->")

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
        `MAG`

        Notes
        -----


        Examples
        --------

        References
        ----------
        [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
        Journal of Machine Learning Research 9 (2008): 1437-1474.
        """
        # # TODO(@daehyun99): [#2385] Implement code logic and test code When Refactor DAG
        # # TODO(@daehyun99): [#2385] Fix Docs (Unify Docs Format)
        # # TODO(@daehyun99): [#2385] Apply type hint(input, output)
        raise NotImplementedError("`has_almost_directed_cycle` is not supported now")

    def _check_new_unshielded_collider(self, u: Hashable, v: Hashable) -> bool:
        """
        Tests if orienting an undirected edge u - v as u -> v creates new unshielded V-structures in the PDAG.

        Checks whether v has any directed parents other than u that are not adjacent to u.

        Returns
        -------
        True, if the orientation u -> v would lead to creation of a new V-structure.
        False, if no new V-structures are formed.
        """
        for node in self.get_parents(v):
            if (node != u) and (not self.has_edge(u, node)):
                return True
        return False

    # ----------------------------------------------------------------------
    # Internal Methods (or Private Methods)
    # ----------------------------------------------------------------------
