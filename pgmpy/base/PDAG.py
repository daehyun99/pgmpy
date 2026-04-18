import itertools
from collections.abc import Hashable, Iterable

import networkx as nx

from pgmpy import logger
from pgmpy.base._base import _CoreGraph


class PDAG(_CoreGraph):
    """
    Class for representing PDAGs (also known as CPDAG). PDAGs are the equivalence classes of
    DAGs and contain both directed and undirected edges.

    Note: In this class, undirected edges are represented using two edges in both direction i.e.
    an undirected edge between X - Y is represented using X -> Y and X <- Y.

    Parameters
    ----------
    directed_ebunch: list, array-like of 2-tuples
        List of directed edges in the PDAG.

    undirected_ebunch: list, array-like of 2-tuples
        List of undirected edges in the PDAG.

    latents: list, array-like
        List of nodes which are latent variables.

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
    >>> from pgmpy.base import PDAG
    >>> pdag = PDAG(
    ...     directed_ebunch=[("A", "C"), ("D", "C")],
    ...     undirected_ebunch=[("B", "A"), ("B", "D")],
    ...     latents=["A"],
    ...     roles={"exposures": ["A"], "outcomes": ["C"]},
    ... )
    >>> sorted(pdag.directed_edges)
    [('A', 'C'), ('D', 'C')]
    >>> sorted(pdag.undirected_edges)
    [('B', 'A'), ('B', 'D')]
    >>> pdag.latents
    {'A'}
    >>> pdag.exposures
    {'A'}
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "--"])

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable, Hashable]] = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        latents: set[Hashable] | None = None,
        roles=None,
    ):
        super().__init__(
            ebunch=ebunch,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

    def chain_component(self, node: Hashable) -> set[Hashable]:
        """
        Returns the chain component of `node`, i.e. all nodes reachable
        through undirected edges.
        """
        visited: set[Hashable] = set()
        to_visit: set[Hashable] = {node}

        while to_visit:
            current = to_visit.pop()
            visited.add(current)
            to_visit |= self.get_reachable_nodes(current, "--") - visited

        return visited

    def has_semidirected_path(
        self,
        u: Hashable,
        v: Hashable,
        blocked_nodes: Iterable[Hashable] | None = None,
        ignore_direct_edge: bool = False,
    ) -> bool:
        """
        Returns True if there exists a semi-directed path from `source` to `target`.

        Semi-directed paths follow directed edges in their forward direction and
        can traverse undirected edges in either direction.

        Parameters
        ----------
        source, target: hashable
            The endpoints of the path.

        blocked_nodes: iterable, optional
            Nodes that are not allowed on the path.

        ignore_direct_edge: bool
            If True, ignores the direct edge `source -> target` when checking for
            a path.
        """
        blocked_nodes = set() if blocked_nodes is None else set(blocked_nodes)

        if (u in blocked_nodes) or (v in blocked_nodes):
            return False

        allowed_nodes = set(self.nodes()) - blocked_nodes
        graph = nx.DiGraph()
        graph.add_nodes_from(allowed_nodes)

        for x, y, edge_type in self.get_edges(data=True):
            if x not in allowed_nodes or y not in allowed_nodes:
                continue

            if edge_type == "->":
                graph.add_edge(x, y)
            elif edge_type == "<-":
                graph.add_edge(y, x)
            elif edge_type == "--":
                graph.add_edge(x, y)
                graph.add_edge(y, x)

        if ignore_direct_edge and graph.has_edge(u, v):
            graph.remove_edge(u, v)

        if not graph.has_node(u) or not graph.has_node(v):
            return False

        return nx.has_path(graph, u, v)

    def has_acyclic_extension(self) -> bool:
        """
        Returns True if the PDAG admits an acyclic DAG extension.
        """
        if not self.undirected_edges:
            return nx.is_directed_acyclic_graph(nx.DiGraph(self.edges()))

        dag = self.to_dag()
        return nx.is_directed_acyclic_graph(dag)

    def is_clique(self, nodes: Iterable) -> bool:
        """
        Checks if a set of nodes forms a clique. A clique is a subgraph
        where every pair of nodes is connected by an edge (fully connected).

        Parameters
        ----------
        nodes: Iterable
            The set of nodes to be checked for clique formation.
        """
        for node1, node2 in itertools.combinations(nodes, 2):
            if not self.has_undirected_edge(node1, node2):
                return False
        return True

    def apply_meeks_rules(self, apply_r4=False, inplace=False, debug=False):
        """
        Applies the Meek's rules to orient the undirected edges of a PDAG to return a CPDAG.

        Parameters
        ----------
        apply_r4: boolean (default=False)
            If True, applies Rules 1 - 4 of Meek's rules.
            If False, applies only Rules 1 - 3.

        inplace: boolean (default=False)
            If True, the PDAG object is modified inplace, otherwise a new modified copy is returned.

        debug: boolean (default=False)
            If True, prints the rules being applied to the PDAG.

        Returns
        -------
        None or pgmpy.base.PDAG: The modified PDAG object.
            If inplace=True, returns None and the object itself is modified.
            If inplace=False, returns a PDAG object.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(
        ...     directed_ebunch=[("A", "B")], undirected_ebunch=[("B", "C"), ("C", "B")]
        ... )
        >>> pdag = pdag.apply_meeks_rules()
        >>> sorted(pdag.directed_edges)
        [('A', 'B'), ('B', 'C')]
        """
        if inplace:
            pdag = self
        else:
            pdag = self.copy()

        changed = True
        while changed:
            changed = False

            # Rule 1: If X -> Y - Z and
            #            (X not adj Z) and
            #            (adding Y -> Z doesn't create cycle) and
            #            (adding Y -> Z doesn't create an unshielded collider) =>  Y → Z
            for y in pdag.nodes():
                # Select x's such that there are directed edges x -> y.
                for x in pdag.get_parents(y):
                    for z in pdag.get_neighbors(y, "--"):
                        if (
                            (not pdag.has_edge(x, z))
                            and (not pdag._check_new_unshielded_collider(y, z))
                            and (not pdag.has_direct_path(z, y))
                        ):
                            pdag.orient_undirected_edge(y, z, inplace=True)
                            changed = True
                            if debug:
                                logger.info(f"Applying Rule 1: {x} -> {y} - {z} => {x} -> {y} -> {z}")

            # Rule 2: If X -> Z -> Y  and X - Y =>  X → Y
            for z in pdag.nodes():
                xs = pdag.get_parents(z)
                ys = pdag.get_children(z)

                for x in xs:
                    for y in ys:
                        if pdag.has_edge(x, y) and pdag.get_edge_type(x, y, 0) == "--":
                            pdag.orient_undirected_edge(x, y, inplace=True)
                            changed = True
                            if debug:
                                logger.info(f"Applying Rule 2: {x} -> {z} -> {y} and {x} - {y} => {x} -> {y}")

            # Rule 3: If X - {Y, Z, W} and {Z, Y} -> W => X -> W
            for x in pdag.nodes():
                undirected_nbs = pdag.get_neighbors(x, "--")

                if len(undirected_nbs) < 3:
                    continue

                for y, z, w in itertools.permutations(undirected_nbs, 3):
                    if pdag.get_edge_type(y, w, 0) == "->" and pdag.get_edge_type(z, w, 0) == "->":
                        pdag.orient_undirected_edge(x, w, inplace=True)
                        changed = True
                        if debug:
                            logger.info(f"Applying Rule 3: {x} - {y}, {z}, {w} {y}, {z} -> {w} => {x} -> {w}")
                        break

            # Rule 4: If d -> c -> b & a - {b, c, d} and b not adj d => a -> b
            if apply_r4:
                for c in pdag.nodes():
                    for b in pdag.get_children(c):
                        for d in pdag.get_parents(c):
                            if b == d or pdag.has_edge(b, d):
                                continue  # b adjacent d => rule not applicable

                            # find nodes a that are undirected neighbor to b, d,
                            #  and directed or undirected neighbor to c
                            cand = set(pdag.get_neighbors(b, "--")).intersection(
                                pdag.get_neighbors(c),
                                pdag.get_neighbors(d, "--"),
                            )
                            for a in cand:
                                pdag.orient_undirected_edge(a, b, inplace=True)
                                changed = True
                                break
        if not inplace:
            return pdag

    def calibrate_directed_undirected_edges(self) -> None:
        """
        Iterates through existing edges to correctly assign directed
        and undirected edges.
        """
        all_edges = set(self.edges)
        undirected = set()
        directed = set()
        for u, v in all_edges:
            if (v, u) in all_edges:
                if u > v:
                    undirected.add((u, v))
            else:
                directed.add((u, v))

        self.undirected_edges = undirected
        self.directed_edges = directed

    def to_cpdag(self):
        """
        Returns the CPDAG corresponding to one DAG extension of the PDAG.
        """
        from pgmpy.base import DAG

        if self.undirected_edges:
            dag = self.to_dag()
        else:
            dag = DAG()
            dag.add_nodes_from(self.nodes())
            dag.add_edges_from(self.edges())
            dag.latents = self.latents

        cpdag = dag.to_pdag()
        for role, vars in self.get_role_dict().items():
            cpdag.with_role(role=role, variables=vars, inplace=True)

        return cpdag

    def to_dag(self):
        """
        Returns one possible DAG which is represented using the PDAG.

        Returns
        -------
        pgmpy.base.DAG: Returns an instance of DAG.

        Examples
        --------
        >>> pdag = PDAG(
        ...     directed_ebunch=[("A", "B"), ("C", "B")],
        ...     undirected_ebunch=[("C", "D"), ("D", "A")],
        ... )
        >>> dag = pdag.to_dag()
        >>> sorted(dag.edges())
        [('A', 'B'), ('C', 'B'), ('D', 'A'), ('D', 'C')]

        References
        ----------
        [1] Dor, Dorit, and Michael Tarsi.
          "A simple algorithm to construct a consistent extension of a partially oriented graph."
            Technicial Report R-185, Cognitive Systems Laboratory, UCLA (1992): 45.
        """
        # Add required edges if it doesn't form a new v-structure or an opposite edge
        # is already present in the network.
        from pgmpy.base import DAG

        dag = DAG()
        # Add all the nodes and the directed edges
        dag.add_nodes_from(self.nodes())
        ebunch = self.get_edges(data=True)
        directed_edges = []
        for u, v, edge_type in ebunch:
            if edge_type == "->":
                directed_edges.append((u, v))
            elif edge_type == "<-":
                directed_edges.append((v, u))
        dag.add_edges_from(directed_edges)
        dag.latents = self.latents

        pdag = self.copy()
        while pdag.number_of_nodes() > 0:
            # find node with (1) no directed outgoing edges and
            #                (2) the set of undirected neighbors is either empty or
            #                    undirected neighbors + parents of X are adjacent
            found = False
            for X in sorted(pdag.nodes()):
                undirected_neighbors = pdag.get_neighbors(X, "--")
                neighbors_are_adjacent = all(
                    pdag.has_edge(Y, Z) or pdag.has_edge(Z, Y)
                    for Z in pdag.get_neighbors(X)
                    for Y in undirected_neighbors
                    if not Y == Z
                )

                if not pdag.get_children(X) and (not undirected_neighbors or neighbors_are_adjacent):
                    found = True
                    # add all edges of X as outgoing edges to dag
                    for Y in pdag.get_neighbors(X, "--"):
                        dag.add_edge(Y, X)
                    pdag.remove_node(X)
                    break

            if not found:
                logger.warning(
                    "PDAG has no faithful extension (= no oriented DAG with the "
                    + "same v-structures as PDAG). Remaining undirected PDAG edges "
                    + "oriented arbitrarily."
                )
                for X, Y in pdag.edges():
                    if not dag.has_edge(Y, X):
                        try:
                            dag.add_edge(X, Y)
                        except ValueError:
                            pass
                break
        return dag

    def to_graphviz(self) -> object:
        """
        Retuns a pygraphviz object for the DAG. pygraphviz is useful for
        visualizing the network structure.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/alarm")
        >>> model.to_graphviz()  # doctest: +ELLIPSIS
        <AGraph ...
        """
        return nx.nx_agraph.to_agraph(self)
