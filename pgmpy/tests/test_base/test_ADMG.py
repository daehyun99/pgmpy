import pytest

from pgmpy.base.ADMG import ADMG
from pgmpy.base.DAG import DAG


class TestADMGInitialization:
    """Test ADMG initialization and basic setup."""

    def test_empty_initialization(self):
        """Test creating an empty ADMG."""
        admg = ADMG()
        assert len(admg.nodes) == 0
        assert len(admg.edges) == 0
        assert len(admg.latents) == 0
        assert len(admg.get_roles()) == 0

    def test_initialization_with_directed_edges(self):
        """Test initialization with directed edges."""
        directed_edges = [("A", "B"), ("B", "C")]
        admg = ADMG(ebunch=directed_edges)

        assert "A" in admg.nodes
        assert "B" in admg.nodes
        assert "C" in admg.nodes
        assert admg.has_edge("A", "B")
        assert admg.has_edge("B", "C")

    def test_initialization_with_bidirected_edges(self):
        """Test initialization with bidirected edges."""
        bidirected_edges = [("X", "Y", "<>"), ("Y", "Z", "<>")]
        admg = ADMG(ebunch=bidirected_edges)

        assert "X" in admg.nodes
        assert "Y" in admg.nodes
        assert "Z" in admg.nodes
        # Bidirected edges create edges in both directions
        assert admg.has_edge("X", "Y")
        assert admg.has_edge("Y", "X")

    def test_initialization_with_latents(self):
        """Test initialization with latent variables."""
        bidirected_edges = [("L1", "L2", "<>")]
        latents = ["L1", "L2"]
        admg = ADMG(ebunch=bidirected_edges, latents=latents)

        assert admg.latents == {"L1", "L2"}

    def test_initialization_with_roles(self):
        """Test initialization with roles variables."""
        directed_edges = [("A", "C"), ("B", "C")]
        roles = {"exposure": ("A", "B"), "outcome": ["C"]}
        admg = ADMG(ebunch=directed_edges, roles=roles)

        assert set(admg.get_role("exposure")) == set(["A", "B"])
        assert admg.get_role("outcome") == ["C"]
        assert set(admg.get_roles()) == set(["exposure", "outcome"])
        assert admg.get_role_dict() == {"exposure": ["A", "B"], "outcome": ["C"]}

    def test_latents_with_role(self):
        edges = [
            ("X", "Y", "->"),
            ("A", "B", "<>"),
            ("B", "C", "<>"),
            ("C", "D", "<>"),
            ("D", "E", "<>"),
            ("E", "F", "<>"),
        ]
        admg = ADMG(
            ebunch=edges,
            latents=["A"],
            roles={"exposure": "X", "outcome": "Y", "latents": "B"},
        )
        admg.with_role(role="latents", variables="C", inplace=True)
        admg.with_role(role="latents", variables=["D", "E", "F"], inplace=True)

        assert admg.latents == {"A", "B", "C", "D", "E", "F"}
        assert set(admg.get_role("latents")) == {"A", "B", "C", "D", "E", "F"}

        with pytest.raises(ValueError, match="Variable 'G' not found in the graph."):
            admg.with_role(role="latents", variables="G", inplace=True)

    def test_latents_without_role(self):
        edges = [
            ("X", "Y", "->"),
            ("A", "B", "<>"),
            ("B", "C", "<>"),
            ("C", "D", "<>"),
            ("D", "E", "<>"),
            ("E", "F", "<>"),
        ]
        admg = ADMG(
            ebunch=edges,
            latents=["A", "B", "C"],
            roles={"exposure": "X", "outcome": "Y", "latents": ("D", "E", "F")},
        )

        admg.without_role(role="latents", variables="A", inplace=True)
        admg.without_role(
            role="latents", variables=["B", "C", "D", "E", "F"], inplace=True
        )

        assert admg.latents == set()
        assert set(admg.get_role("latents")) == set()


class TestADMGNodeOperations:
    """Test node addition and validation."""

    def test_add_node(self):
        """Test adding a single node."""
        admg = ADMG()
        admg.add_node("A")
        admg.add_node("B")

        assert set(admg.nodes()) == {"A", "B"}

    def test_add_nodes_from(self):
        """Test adding multiple nodes at once."""
        admg = ADMG()
        admg.add_nodes_from(["A", "B"])
        admg.add_nodes_from(set(["C", "D"]))

        assert set(admg.nodes()) == {"A", "B", "C", "D"}


class TestADMGEdgeOperations:
    """Test edge addition and validation."""

    def test_add_directed_edge(self):
        """Test adding directed edges."""
        admg = ADMG()
        egdes = [("A", "B", "->"), ("B", "C", "->")]
        admg.add_edges_from(egdes)

        assert admg.has_edge("A", "B")
        assert set(admg.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}

    def test_add_bidirected_edge(self):
        """Test adding bidirected edges."""
        admg = ADMG()
        admg.add_edge("X", "Y", "<>")

        assert admg.has_edge("X", "Y")
        assert set(admg.get_edges(data=True)) == {("X", "Y", "<>")}

    def test_add_directed_edges(self):
        """Test adding multiple directed edges at once."""
        admg = ADMG()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "->")]
        admg.add_edges_from(edges)

        for u, v, _ in edges:
            assert admg.has_edge(u, v)

        assert set(admg.get_edges(data=True)) == set(edges)

    def test_add_bidirected_edgess_batch(self):
        """Test adding multiple bidirected edges at once."""
        admg = ADMG()
        edges = [("X", "Y", "<>"), ("Y", "Z", "<>")]
        admg.add_edges_from(edges)

        for u, v, _ in edges:
            assert admg.has_edge(u, v)

        assert set(admg.get_edges(data=True)) == set(edges)

    def test_cycle_detection(self):
        """Test that cycles are prevented in directed edges."""
        admg = ADMG()
        admg.add_edge("A", "B", "->")
        admg.add_edge("B", "C", "->")

        # This should raise an error as it creates a cycle
        with pytest.raises(ValueError, match="Adding this edge would create a cycle"):
            admg.add_edge("C", "A", "->")

    def test_none_node_rejection(self):
        """Test that None nodes are rejected."""
        admg = ADMG()

        with pytest.raises(ValueError):
            admg.add_edge(None, "B", "->")

        with pytest.raises(ValueError):
            admg.add_edge("A", None, "<>")

    def test_self_bidirected_edge_rejection(self):
        """Test that self-loops in bidirected edges are rejected."""
        admg = ADMG()

        with pytest.raises(ValueError):
            admg.add_edge("A", "A", "<>")


class TestADMGRelationships:
    """Test getting parents, children, spouses, etc."""

    def setup_method(self):
        """Set up a test graph for relationship tests."""
        self.admg = ADMG()
        edges = [  # Directed, Bidirected
            ("A", "B", "->"),
            ("B", "C", "->"),
            ("D", "B", "->"),
            ("A", "D", "<>"),
            ("B", "E", "<>"),
        ]

        self.admg.add_edges_from(edges)

    def test_get_parents(self):
        """Test get_parents of nodes."""
        parents = self.admg.get_parents("B")

        assert "A" in parents
        assert "D" in parents
        assert len(parents) == 2

    def test_get_children(self):
        """Test getting children of nodes."""
        children = self.admg.get_children("B")

        assert "C" in children
        assert len(children) == 1

    def test_get_spouses(self):
        """Test getting spouses (bidirected connections)."""
        spouses_a = self.admg.get_spouses("A")
        spouses_b = self.admg.get_spouses("B")

        assert "D" in spouses_a
        assert "E" in spouses_b

    def test_get_ancestors(self):
        """Test getting ancestors."""
        ancestors_c = self.admg.get_ancestors("C")

        assert "A" in ancestors_c
        assert "B" in ancestors_c
        assert "D" in ancestors_c
        assert "C" in ancestors_c  # Node includes itself

    def test_get_descendants(self):
        """Test getting descendants."""
        descendants_a = self.admg.get_descendants("A")

        assert "B" in descendants_a
        assert "C" in descendants_a
        assert "A" in descendants_a  # Node includes itself

    @pytest.mark.skip(
        reason="Refactoring: Skip for evaluation integration into _GraphAlgorithmMixin class. (Related: #2384, #2385)"
    )
    def test_get_district(self):
        """Test getting district (bidirected connected components)."""
        district_a = self.admg.get_district("A")

        # A and D are connected by bidirected edge
        assert "A" in district_a
        assert "D" in district_a

    def test_nonexistent_node_error(self):
        """Test that operations on nonexistent nodes raise errors."""
        with pytest.raises(ValueError):
            self.admg.get_parents("Z")

        with pytest.raises(ValueError):
            self.admg.get_children("Z")


class TestADMGGraphOperations:
    """Test advanced graph operations."""

    def setup_method(self):
        """Set up a test graph."""
        edges = [
            ("A", "B", "->"),
            ("B", "C", "->"),
            ("D", "E", "->"),
            ("A", "D", "<>"),
            ("B", "E", "<>"),
        ]
        self.admg = ADMG()

        self.admg.add_edges_from(edges)
        self.admg.add_node("F", latent=True)

        self.admg.with_role(role="exposure", variables={"A"}, inplace=True)
        self.admg.with_role(role="outcome", variables={"C"}, inplace=True)

    @pytest.mark.skip(
        reason="Refactoring: Skip for evaluation integration into _GraphAlgorithmMixin class. (Related: #2384, #2385)"
    )
    def test_get_ancestral_graph(self):
        """Test getting ancestral graph of a subset of nodes."""
        ancestral = self.admg.get_ancestral_graph(["A", "B", "D"])

        assert "A" in ancestral.nodes
        assert "B" in ancestral.nodes
        assert "D" in ancestral.nodes
        assert "C" not in ancestral.nodes
        assert "E" not in ancestral.nodes

        # Should have directed edge A -> B
        assert ancestral.has_edge("A", "B")
        # Should have bidirected edge A <-> D
        assert ancestral.has_edge("A", "D")
        assert ancestral.has_edge("D", "A")

    def test_get_ancestral_graph_invalid_nodes(self):
        """Test ancestral graph with invalid nodes."""
        with pytest.raises(ValueError, match="Input nodes must be subset"):
            self.admg.get_ancestral_graph(["A", "Z"])

    @pytest.mark.skip(
        reason="Refactoring: Skip for evaluation integration into _GraphAlgorithmMixin class. (Related: #2384, #2385)"
    )
    def test_get_markov_blanket(self):
        """Test getting Markov blanket."""
        mb_b = self.admg.get_markov_blanket("B")

        # B's Markov blanket should include its parents, children, and spouses
        assert "A" in mb_b  # parent
        assert "C" in mb_b  # child
        assert "E" in mb_b  # spouse

    @pytest.mark.skip(
        reason="Refactoring: Skip for evaluation integration into _GraphAlgorithmMixin class. (Related: #2384, #2385)"
    )
    def test_to_dag(self):
        """Test conversion to DAG."""
        dag = self.admg.to_dag()

        # Should return a pgmpy DAG instance
        from pgmpy.base.DAG import DAG as pgmpy_DAG

        assert isinstance(dag, pgmpy_DAG)

    def test_admg_equality(self):
        """
        Test the `__eq__` method
        which compares both graph structure and variable-role mappings to allow comparison of two models.
        """
        # Case1: When the models are the same
        other1 = ADMG(
            ebunch=[
                ("A", "B", "->"),
                ("B", "C", "->"),
                ("D", "E", "->"),
                ("A", "D", "<>"),
                ("B", "E", "<>"),
            ],
            latents=["D"],
            roles={"exposure": ["A"], "outcome": ["C"]},
        )

        # Case2: When the models differ (DAG)
        other2 = DAG(
            ebunch=[("A", "C", "->"), ("D", "C", "->")],
            latents=["D"],
            roles={"exposure": "A", "adjustment": "D", "outcome": "C"},
        )

        # Case3: When the directed edges differ
        other3 = ADMG(
            ebunch=[
                ("A", "C", "->"),
                ("B", "C", "->"),
                ("D", "E", "->"),
                ("A", "D", "<>"),
                ("B", "E", "<>"),
            ],
            latents=["D"],
            roles={"exposure": ["A"], "outcome": ["C"]},
        )

        # Case4: When the bidirected edges differ
        other4 = ADMG(
            ebunch=[
                ("A", "B", "->"),
                ("B", "C", "->"),
                ("D", "E", "->"),
                ("A", "E", "<>"),
                ("B", "E", "<>"),
            ],
            latents=["D"],
            roles={"exposure": ["A"], "outcome": ["C"]},
        )

        # Case5: When the latents variables differ
        other5 = ADMG(
            ebunch=[
                ("A", "B", "->"),
                ("B", "C", "->"),
                ("D", "E", "->"),
                ("A", "D", "<>"),
                ("B", "E", "<>"),
            ],
            latents=["B"],
            roles={"exposure": ["A"], "outcome": ["C"]},
        )

        # Case6: When the roles variables differ
        other6 = ADMG(
            ebunch=[
                ("A", "B", "->"),
                ("B", "C", "->"),
                ("D", "E", "->"),
                ("A", "D", "<>"),
                ("B", "E", "<>"),
            ],
            latents=["D"],
            roles={"exposure": ["A"], "adjustment": "D", "outcome": ["C"]},
        )

        assert self.admg.__eq__(other1) is True
        assert self.admg.__eq__(other2) is False
        assert self.admg.__eq__(other3) is False
        assert self.admg.__eq__(other4) is False
        assert self.admg.__eq__(other5) is False
        assert self.admg.__eq__(other6) is False


class TestADMGSeparation:
    """Test m-separation and m-connection."""

    def setup_method(self):
        """Set up a test graph for separation tests."""
        self.admg = ADMG()
        self.admg.add_edges_from(
            [("A", "C", "->"), ("B", "C", "->"), ("C", "D", "->"), ("A", "B", "<>")]
        )

    @pytest.mark.skip(
        reason="Refactoring: Skip for evaluation integration into _GraphAlgorithmMixin class. (Related: #2384, #2385)"
    )
    def test_is_m_separated(self):
        """Test m-separation check."""
        # A and B should not be m-separated (they have bidirected edge)
        assert not self.admg.is_mseparated("A", "B")

        # Test with conditional set
        assert self.admg.is_mseparated("A", "D", conditional_set={"C"}) is True
        assert self.admg.is_mseparated("A", "D", conditional_set=set()) is False
        # This depends on the specific graph structure and d-separation rules

    @pytest.mark.skip(
        reason="Refactoring: Skip for evaluation integration into _GraphAlgorithmMixin class. (Related: #2384, #2385)"
    )
    def test_is_m_connected(self):
        """Test m-connection check."""
        # This should be the opposite of m-separation
        connected = self.admg.is_mconnected("A", "B")
        separated = self.admg.is_mseparated("A", "B")
        assert connected != separated
