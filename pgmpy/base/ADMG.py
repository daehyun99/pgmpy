from typing import Hashable, Iterable, Optional

from pgmpy.base._base import _CoreGraph
from pgmpy.base.DAG import DAG as pgmpy_DAG


class ADMG(_CoreGraph):
    """
    A class representing an Acyclic Directed Mixed Graph (ADMG).

    An ADMG is a directed graph that allows for both directed and bidirected edges.
    This class extends the `networkx.MultiDiGraph` and provides additional functionality
    for operations involving directed and bidirected edges.

    Parameters
    ----------
    directed_ebunch : list of tuple, optional
        List of directed edges to initialize the graph, where each tuple is (u, v).
    bidirected_ebunch : list of tuple, optional
        List of bidirected edges to initialize the graph, where each tuple is (u, v).
    latents : set of str, optional
        Set of latent variables in the graph. These are not directly represented as nodes
        but are used to indicate the presence of bidirected edges.
    roles : dict, optional (default: None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).
        If provided, this will automatically assign roles to the nodes in the graph.
        Passing a key-value pair via ``roles`` is equivalent to calling
        ``with_role(role, variables)`` for each key-value pair in the dictionary.

    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "<>"])

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

    def add_edge(self, u, v, edge_type="->", key=None, **kwargs):
        # No additional comments are needed, as the comments in _CoreGraph are utilized.

        # Need Logic of Checking cycle.

        return super().add_edge(u, v, edge_type, key, **kwargs)

    def add_edges_from(self, ebunch, **kwargs):
        # No additional comments are needed, as the comments in _CoreGraph are utilized.
        self._validate_edges(ebunch=ebunch)
        for edge in ebunch:
            if len(edge) == 3:
                u, v, edge_type = edge
                self.add_edge(u, v, edge_type=edge_type, **kwargs)
            elif len(edge) == 4:
                u, v, key, edge_type = edge
                self.add_edge(u, v, edge_type=edge_type, key=key, **kwargs)

    def get_district(self, nodes):
        """
        Return district of a node: maximal set connected via bidirected edges.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Nodes in the same bidirected-connected component.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)
        components = set()

        for node in nodes_set:
            component = self.get_reachable_nodes(node, edge_type="<>")
            components.update(component)

        return components

    def get_ancestral_graph(self, nodes):
        """
        Return the ancestral graph induced by the input nodes.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes to induce subgraph on.

        Returns
        -------
        ADMG
            Subgraph induced by ancestors of the given nodes.

        Raises
        ------
        ValueError
            If any input node is not in the graph.
        """
        nodes_set = {nodes} if isinstance(nodes, str) else set(nodes)

        ancestors = set(nodes_set)
        for node in nodes_set:
            ancestor = self.get_ancestors(node)
            ancestors.update(ancestor)

        new_admg = ADMG()
        new_admg.add_nodes_from(ancestors)

        for u, v, key, data in self.edges(keys=True, data=True):
            if (u in ancestors) and (v in ancestors):
                new_admg.add_edge(u, v, edge_type=data, key=key)

        return new_admg

    def get_markov_blanket(self, nodes):
        """
        Compute the Markov blanket for the given node(s).

        Includes:
        - Parents
        - Children
        - Spouses (nodes sharing a child)
        - Parents of nodes in the district

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Set of nodes in the Markov blanket.
        """
        nodes_set = {nodes} if isinstance(nodes, set) else set(nodes)
        if not nodes_set.issubset(self.nodes):
            raise ValueError("Input nodes must be subset of graph's nodes.")
        markov_blanket = set()
        for node in nodes_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} is not in the graph.")
            # Get parents
            parents = self.get_directed_parents(node)
            district_parents = self.get_bidirected_parents(node)
            markov_blanket.update(parents)
            markov_blanket.update(district_parents)
            # Get children
            children = self.get_children(node)
            markov_blanket.update(children)
            # Get spouses
            spouses = self.get_spouses(node)
            markov_blanket.update(spouses)
        return markov_blanket

    def to_dag(self):
        """
        Project ADMG into a DAG by introducing latent variables for bidirected edges.

        Returns
        -------
        pgmpy.base.DAG.DAG
            DAG with latent variables replacing bidirected edges.
        """
        dag_edges = []

        # Add directed edges
        for u, v, data in self.edges(data=True):
            if data.get("type") == "directed":
                dag_edges.append((u, v))

        # add latent nodes and edges for bidirected edges
        latent_nodes_map = {}
        for u, v, data in self.edges(data=True):
            if data.get("type") == "bidirected":
                sorted_pair = tuple(sorted((u, v)))
                if sorted_pair not in latent_nodes_map:
                    latent_var = f"L_{sorted_pair[0]}_{sorted_pair[1]}"
                    latent_nodes_map[sorted_pair] = latent_var
                    dag_edges.append((latent_var, sorted_pair[0]))
                    dag_edges.append((latent_var, sorted_pair[1]))

        dag_nodes = set(self.nodes()) | set(latent_nodes_map.values())

        # Create a new DAG instance
        dag_instance = pgmpy_DAG()
        dag_instance.add_nodes_from(dag_nodes)
        dag_instance.add_edges_from(dag_edges)

        return dag_instance

    def is_mseparated(
        self,
        nodes_u,
        nodes_v,
        conditional_set=None,
    ):
        """
        Test m-separation between two sets of nodes given a conditioning set.

        Parameters
        ----------
        nodes_u : str or iterable of str
            First set of nodes.

        nodes_v : str or iterable of str
            Second set of nodes.

        conditional_set : set of str, optional
            Conditioning set (default is empty set).

        Returns
        -------
        bool
            True if nodes_u and nodes_v are m-separated; False otherwise.
        """
        if conditional_set is None:
            conditional_set = set()

        # Convert nodes_u and nodes_v to sets
        nodes_u_set = {nodes_u} if isinstance(nodes_u, str) else set(nodes_u)
        nodes_v_set = {nodes_v} if isinstance(nodes_v, str) else set(nodes_v)

        new_dag = self.to_dag()
        for u in nodes_u_set:
            for v in nodes_v_set:
                # if they are dconnected, they are not mseparated
                if new_dag.is_dconnected(u, v, observed=conditional_set):
                    return False
        return True

    def is_mconnected(
        self,
        nodes_u,
        nodes_v,
        conditional_set=None,
    ):
        """
        Test m-connectedness between two node sets given a conditioning set.

        Parameters
        ----------
        nodes_u : str or iterable of str
            First set of nodes.

        nodes_v : str or iterable of str
            Second set of nodes.

        conditional_set : set of str, optional
            Conditioning set.

        Returns
        -------
        bool
            True if m-connected; False if m-separated.
        """
        return not self.is_mseparated(nodes_u, nodes_v, conditional_set)

    def mconnected_nodes(self, nodes_u, nodes_v=None, conditional_set=None):
        """
        Find all nodes m-connected to nodes in `nodes_u` given `conditional_set`.

        Parameters
        ----------
        nodes_u : str or iterable of str
            Set of source nodes.

        nodes_v : str or iterable of str, optional
            If provided, filters the result to this set.

        conditional_set : set of str, optional
            Conditioning set (default is empty set).

        Returns
        -------
        set
            Nodes m-connected to `nodes_u` (or their intersection with `nodes_v` if provided).
        """
        if conditional_set is None:
            conditional_set = set()

        dag = self.to_dag()
        if isinstance(nodes_u, str):
            nodes_u = [nodes_u]

        m_connected_set = set()

        for node in nodes_u:
            active_nodes = dag.active_trail_nodes(node, observed=conditional_set)
            active_nodes = {n for n in active_nodes if not str(n).startswith("L_")}
            m_connected_set.update(active_nodes)

        if nodes_v is not None:
            nodes_v_set = {nodes_v} if isinstance(nodes_v, str) else set(nodes_v)
            return m_connected_set & nodes_v_set

        return m_connected_set
