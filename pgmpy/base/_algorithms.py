#!/usr/bin/env python3

from collections.abc import Hashable, Iterable

import networkx as nx

from pgmpy.utils.types import Self


class _GraphAlgorithms:
    """Graph-algorithm methods for ``_CoreGraph``-based classes (inherited, not instantiated alone)."""

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
        >>> sorted(ancestral_graph.get_edges(data=True))
        [('A', 'B', '->'), ('B', 'C', '->')]

        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)

        ancestors = set(nodes_set)
        for node in nodes_set:
            ancestor = self.get_ancestors(node)
            ancestors.update(ancestor)

        return self.get_subgraph(ancestors)

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
        >>> from pgmpy.base import ADMG
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

        >>> sorted(admg.get_markov_blanket("B"))
        ['A', 'C', 'E']

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
        >>> graph = _CoreGraph(edge_list=edges)
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

    def has_path(self, source: Hashable, target: Hashable, edge_types: str | Iterable[str] | None = None) -> bool:
        """
        Check whether a path from `source` to `target` exists, following only edges of `edge_types`.

        Each step of the path follows an edge whose type (read from the current node's endpoint) is in
        `edge_types`; e.g. ``edge_types="->"`` checks for a directed path. ``None`` follows any edge
        type. A node always has a (trivial) path to itself.

        Parameters
        ----------
        source, target : Hashable
            The endpoints of the path. Both must be present in the graph.

        edge_types : str or iterable of str, optional (default: None)
            The edge type(s) a step may follow; ``None`` allows any type.

        Returns
        -------
        bool

        See Also
        --------
        get_all_paths : Enumerate the paths themselves.
        get_reachable_nodes : All nodes reachable under the given edge types.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->")])
        >>> graph.has_path("A", "C", edge_types="->")
        True

        """
        if target not in self.nodes():
            raise ValueError(f"Node {target} not in graph.")
        return target in self.get_reachable_nodes(source, edge_types)

    def get_all_paths(
        self, source: Hashable, target: Hashable, edge_types: str | Iterable[str] | None = None
    ) -> list[list[Hashable]]:
        """
        Return all simple paths (no repeated nodes) from `source` to `target` following `edge_types`.

        Each step follows an edge whose type (read from the current node's endpoint) is in
        `edge_types`; e.g. ``edge_types="->"`` enumerates directed paths. ``None`` follows any edge
        type. If ``source == target`` the single trivial path ``[source]`` is returned.

        Parameters
        ----------
        source, target : Hashable
            The endpoints of the paths. Both must be present in the graph.

        edge_types : str or iterable of str, optional (default: None)
            The edge type(s) a step may follow; ``None`` allows any type.

        Returns
        -------
        paths : list of lists
            Each inner list is a path given as a sequence of nodes.

        See Also
        --------
        has_path : Whether any such path exists.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->"), ("A", "C", "->")])
        >>> sorted(graph.get_all_paths("A", "C", edge_types="->"))
        [['A', 'B', 'C'], ['A', 'C']]

        """
        for node in (source, target):
            if node not in self.nodes():
                raise ValueError(f"Node {node} not in graph.")

        paths = []
        stack = [(source, [source])]
        while stack:
            current, path = stack.pop()
            if current == target:
                paths.append(path)
                continue
            for neighbor in self.get_neighbors(current, edge_types):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
        return paths

    def get_topological_order(self) -> list[Hashable]:
        """
        Return a topological ordering of the nodes consistent with the directed edges.

        A topological order is an ordering of the nodes in which every directed edge ``u -> v`` has
        ``u`` before ``v``. Following the standard definition for mixed graphs [1]_, bidirected
        (``"<>"``) and undirected (``"--"``) edges impose *no* ordering constraint: a bidirected
        edge encodes latent confounding (no ancestral relation), and undirected edges form chain
        components that are ordered only through their directed edges. The order is therefore well
        defined for DAGs, ADMGs, and (C)PDAGs/MAGs, and is computed on the directed sub-graph.

        Circle endpoints (``"o"``, as used in PAGs) leave the orientation -- and hence the ancestral
        order -- genuinely uncertain, so a ``ValueError`` is raised if the graph contains any circle
        mark.

        Returns
        -------
        order : list
            The nodes in a topological order.

        Raises
        ------
        ValueError
            If the graph contains any edge with a circle endpoint (i.e. a PAG).

        References
        ----------
        .. [1] Richardson, T. S. (2003). Markov Properties for Acyclic Directed Mixed Graphs.
               Scandinavian Journal of Statistics, 30(1), 145-157.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->")])
        >>> graph.get_topological_order()
        ['A', 'B', 'C']
        """
        # Circle endpoints (PAGs) survive canonicalization only as "o-", "o>" and "oo".
        circle_types = self.get_unique_edge_types() & {"o-", "o>", "oo"}
        if circle_types:
            raise ValueError(
                "Topological order is undefined for graphs with circle endpoints (PAGs), where edge "
                f"orientation is uncertain. Found circle edge types: {sorted(circle_types)}."
            )
        dag = nx.DiGraph()
        dag.add_nodes_from(self.nodes())
        dag.add_edges_from((u, v) for u, v, _ in self.get_edges(edge_types={"->"}))
        return list(nx.topological_sort(dag))

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
        # networkx_edge_list = super().edges(keys=True, data=True)
        # from pgmpy.base import DAG
        # dag = DAG()
        # for edge in networkx_edge_list:
        # if edge[-1] == {edge[0]: "-", edge[1]: ">"}:
        # dag.add_edge(edge[0], edge[1], "->")

        dag = nx.DiGraph()
        dag.add_nodes_from(self.nodes())
        dag.add_edges_from((u, v) for u, v, _ in self.get_edges(edge_types={"->"}))
        return not nx.is_directed_acyclic_graph(dag)

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
