import pytest

from pgmpy.base import ADMG, MAG
from pgmpy.base._base import _CoreGraph


@pytest.fixture
def AncestralGraph():
    """
    References
    ----------
    [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
    Journal of Machine Learning Research 9 (2008): 1437-1474. Figure 1-(a).
    """
    edges1 = [  # an ancestral graph that is not maximal.
        ("A", "B", "<>"),
        ("A", "C", "<>"),
        ("B", "D", "<>"),
        ("A", "D", "->"),
        ("B", "C", "->"),
    ]
    graph = _CoreGraph(edge_list=edges1)
    return graph


@pytest.fixture
def MaximalAncestralGraph():
    """
    References
    ----------
    [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
    Journal of Machine Learning Research 9 (2008): 1437-1474. Figure 1-(b).
    """
    edges1 = [  # an ancestral graph that is not maximal.
        ("A", "B", "<>"),
        ("A", "C", "<>"),
        ("B", "D", "<>"),
        ("A", "D", "->"),
        ("B", "C", "->"),
        ("C", "D", "<>"),
    ]
    graph = _CoreGraph(edge_list=edges1)
    return graph


class TestGraphAlgorithmMixin:
    def test_is_collider(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 3.
        """
        graph = _CoreGraph()
        graph.add_edge("T", "M", "->")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_collider("T", "O", "M") == False
        assert graph.is_collider("T", "I", "M") == True
        assert graph.is_collider("T", "B", "M") == True
        assert graph.is_collider("T", "U", "M") == False

        graph = _CoreGraph()
        graph.add_edge("T", "M", "<-")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_collider("T", "O", "M") == False
        assert graph.is_collider("T", "I", "M") == False
        assert graph.is_collider("T", "B", "M") == False
        assert graph.is_collider("T", "U", "M") == False

        graph = _CoreGraph()
        graph.add_edge("T", "M", "<>")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_collider("T", "O", "M") == False
        assert graph.is_collider("T", "I", "M") == True
        assert graph.is_collider("T", "B", "M") == True
        assert graph.is_collider("T", "U", "M") == False

        graph = _CoreGraph()
        graph.add_edge("T", "M", "--")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_collider("T", "O", "M") == False
        assert graph.is_collider("T", "I", "M") == False
        assert graph.is_collider("T", "B", "M") == False
        assert graph.is_collider("T", "U", "M") == False

        graph = _CoreGraph()
        with pytest.raises(ValueError):
            graph.is_collider("T", "O", "M")

    def test_is_collider_neighbors(self):
        graph = _CoreGraph()
        graph.add_edge("T", "M", "->")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        graph.add_edge("T", "I", "->")
        graph.add_edge("T", "B", "<>")

        assert graph.is_collider("T", "O", "M") == False
        assert graph.is_collider("T", "I", "M") == False
        assert graph.is_collider("T", "B", "M") == False
        assert graph.is_collider("T", "U", "M") == False

        graph = _CoreGraph()
        graph.add_edge("T", "M", "<>")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        graph.add_edge("T", "I", "--")
        graph.add_edge("T", "B", "<-")

        assert graph.is_collider("T", "O", "M") == False
        assert graph.is_collider("T", "I", "M") == False
        assert graph.is_collider("T", "B", "M") == False
        assert graph.is_collider("T", "U", "M") == False

    # @pytest.mark.skip(reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class.")
    # def test_is_m_separator(self):
    #     """
    #     References
    #     ----------
    #     [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40. Figure 3.
    #     """
    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "->")

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_separator("T", "O", "M") == True
    #     assert graph.is_m_separator("T", "I", "M") == False
    #     assert graph.is_m_separator("T", "B", "M") == False
    #     assert graph.is_m_separator("T", "U", "M") == True

    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "<-")

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_separator("T", "O", "M") == True
    #     assert graph.is_m_separator("T", "I", "M") == True
    #     assert graph.is_m_separator("T", "B", "M") == True
    #     assert graph.is_m_separator("T", "U", "M") == True

    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "<>")

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_separator("T", "O", "M") == True
    #     assert graph.is_m_separator("T", "I", "M") == False
    #     assert graph.is_m_separator("T", "B", "M") == False
    #     assert graph.is_m_separator("T", "U", "M") == True

    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "--")

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_separator("T", "O", "M") == True
    #     assert graph.is_m_separator("T", "I", "M") == True
    #     assert graph.is_m_separator("T", "B", "M") == True
    #     assert graph.is_m_separator("T", "U", "M") == True

    # @pytest.mark.skip(reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class.")
    # def test_is_m_separator_with_latent(self):
    #     """
    #     References
    #     ----------
    #     [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40. Figure 3.
    #     """
    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "->")
    #     graph.latents = {"M"}

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_separator("T", "O", "M") == False
    #     assert graph.is_m_separator("T", "I", "M") == True
    #     assert graph.is_m_separator("T", "B", "M") == True
    #     assert graph.is_m_separator("T", "U", "M") == False

    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "<-")
    #     graph.latents = {"M"}

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_separator("T", "O", "M") == False
    #     assert graph.is_m_separator("T", "I", "M") == False
    #     assert graph.is_m_separator("T", "B", "M") == False
    #     assert graph.is_m_separator("T", "U", "M") == False

    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "<>")
    #     graph.latents = {"M"}

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_separator("T", "O", "M") == False
    #     assert graph.is_m_separator("T", "I", "M") == True
    #     assert graph.is_m_separator("T", "B", "M") == True
    #     assert graph.is_m_separator("T", "U", "M") == False

    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "--")
    #     graph.latents = {"M"}

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_separator("T", "O", "M") == False
    #     assert graph.is_m_separator("T", "I", "M") == False
    #     assert graph.is_m_separator("T", "B", "M") == False
    #     assert graph.is_m_separator("T", "U", "M") == False

    # @pytest.mark.skip(reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class.")
    # def test_is_m_connected(self):
    #     """
    #     References
    #     ----------
    #     [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40. Figure 3.
    #     """
    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "->")

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_connected("T", "O", "M") == False
    #     assert graph.is_m_connected("T", "I", "M") == True
    #     assert graph.is_m_connected("T", "B", "M") == True
    #     assert graph.is_m_connected("T", "U", "M") == False

    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "<-")

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_connected("T", "O", "M") == False
    #     assert graph.is_m_connected("T", "I", "M") == False
    #     assert graph.is_m_connected("T", "B", "M") == False
    #     assert graph.is_m_connected("T", "U", "M") == False

    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "<>")

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_connected("T", "O", "M") == False
    #     assert graph.is_m_connected("T", "I", "M") == True
    #     assert graph.is_m_connected("T", "B", "M") == True
    #     assert graph.is_m_connected("T", "U", "M") == False

    #     graph = _CoreGraph()
    #     graph.add_edge("T", "M", "--")

    #     graph.add_edge("M", "O", "->")
    #     graph.add_edge("M", "I", "<-")
    #     graph.add_edge("M", "B", "<>")
    #     graph.add_edge("M", "U", "--")

    #     assert graph.is_m_connected("T", "O", "M") == False
    #     assert graph.is_m_connected("T", "I", "M") == False
    #     assert graph.is_m_connected("T", "B", "M") == False
    #     assert graph.is_m_connected("T", "U", "M") == False

    # @pytest.mark.skip(reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class.")
    # def test_is_minimal_m_separator(self):
    #     """
    #     References
    #     ----------
    #     [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40. Figure 1.
    #     """
    #     # TODO(@daehyun99): [#2384] Implement code logic and test code
    #     ...

    # @pytest.mark.skip(reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class.")
    # def test_get_m_separator(self):
    #     """
    #     References
    #     ----------
    #     [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40. Figure 1.
    #     """
    #     # TODO(@daehyun99): [#2384] Implement code logic and test code
    #     ...

    # @pytest.mark.skip(reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class.")
    # def test_get_m_separator_with_latent(self):
    #     """
    #     References
    #     ----------
    #     [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40. Figure 1.
    #     """
    #     # TODO(@daehyun99): [#2384] Implement code logic and test code
    #     ...

    # @pytest.mark.skip(reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class.")
    # def test_get_minimal_m_separator(self):
    #     """
    #     References
    #     ----------
    #     [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40. Figure 1.
    #     """
    #     # TODO(@daehyun99): [#2384] Implement code logic and test code
    #     ...

    # @pytest.mark.skip(reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class.")
    # def test_get_m_separators(self):
    #     """
    #     References
    #     ----------
    #     [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40. Figure 1.
    #     """
    #     # TODO(@daehyun99): [#2384] Implement code logic and test code
    #     ...

    # @pytest.mark.skip(reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class.")
    # def test_get_m_separators_with_latent(self):
    #     """
    #     References
    #     ----------
    #     [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40. Figure 1.
    #     """
    #     # TODO(@daehyun99): [#2384] Implement code logic and test code
    #     ...

    # @pytest.mark.skip(reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class.")
    # def test_get_minimal_m_separators(self):
    #     """
    #     References
    #     ----------
    #     [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
    #     "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
    #     Artificial Intelligence 270 (2019): 1-40. Figure 1.

    #     [2] Takata, Ken.
    #     "Space-optimal, backtracking algorithms to list the minimal vertex separators of a graph."
    #     Discrete Applied Mathematics 158 (2010): 1660-1667. Figure 1.
    #     """
    #     graph = _CoreGraph()
    #     graph.add_nodes_from(["A", "two", "three", "four", "five", "B"])

    #     # TODO(@daehyun99): [#2384] Implement code logic and test code
    #     ...

    def has_directed_cycle(self):
        """
        Testing `has_directed_cycle` method of Graph class(_CoreGraph, DAG, ADMG)
        """
        # TODO(@daehyun99): [#2385] Consider implement `has_directed_cycle` method.

    def has_almost_directed_cycle(self):
        """
        Testing `has_almost_directed_cycle` method of Graph class(_CoreGraph, DAG, ADMG)
        """
        # TODO(@daehyun99): [#2385] Consider implement `has_almost_directed_cycle` method.

    def test_get_ancestral_graph(self):
        """
        Testing `_get_ancestral_graph` method of All graph class
        """
        # _CoreGraph
        graph1 = _CoreGraph()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "--")]
        graph1.add_edges_from(edges)

        new_graph1 = graph1.get_ancestral_graph("C")

        assert isinstance(new_graph1, _CoreGraph)
        assert set(new_graph1.nodes()) == {"A", "B", "C"}
        assert set(new_graph1.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}

        # ADMG
        graph2 = ADMG()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "->")]
        graph2.add_edges_from(edges)

        new_graph2 = graph2.get_ancestral_graph("C")

        assert isinstance(new_graph2, ADMG)
        assert set(new_graph2.nodes()) == {"A", "B", "C"}
        assert set(new_graph2.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}

        # MAG
        graph3 = MAG()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "--")]
        graph3.add_edges_from(edges)

        new_graph3 = graph3.get_ancestral_graph("C")

        assert isinstance(new_graph3, MAG)
        assert set(new_graph3.nodes()) == {"A", "B", "C"}
        assert set(new_graph3.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}

        # TODO(@daehyun99): [#2384] Expand DAG
        ...

        # TODO(@daehyun99): [#2384] Expand PAG
        ...

        # TODO(@daehyun99): [#2384] Expand UndirectedGraph
        ...

        # _CoreGraph with rols
        graph = _CoreGraph()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "--")]
        graph.add_edges_from(edges)
        graph.exposures = "A"
        graph.outcomes = {"C", "D"}
        graph.latents = {"D", "E"}

        new_graph = graph.get_ancestral_graph("C")

        assert isinstance(new_graph, _CoreGraph)
        assert set(new_graph.nodes()) == {"A", "B", "C"}
        assert new_graph.exposures == {"A"}
        assert new_graph.outcomes == {"C"}
        assert new_graph.latents == set()
        assert set(new_graph.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}

    def test_get_markov_blanket(self):
        """Test getting Markov blanket."""
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
        self.admg.with_role(role="exposures", variables={"A"}, inplace=True)
        self.admg.with_role(role="outcomes", variables={"C"}, inplace=True)

        mb_b = self.admg.get_markov_blanket("B")

        assert mb_b == {"A", "C", "E"}

        # TODO: Implement test code for DAG

        # TODO: Implement failing code for other graphs

    def test_get_markov_blanket_fails(self):
        """Test getting Markov blanket."""
        graph = _CoreGraph()
        with pytest.raises(TypeError):
            graph.get_markov_blanket("B")

        graph = MAG()
        with pytest.raises(TypeError):
            graph.get_markov_blanket("B")

        # # TODO(@daehyun99): Activate test code when refactor PDAG
        # graph = PDAG()
        # with pytest.raises(TypeError):
        #     graph.get_markov_blanket("B")

    def test_has_inducing_path(self, AncestralGraph, MaximalAncestralGraph):
        """"""
        graph = AncestralGraph
        assert graph.has_inducing_path("C", "D", set())

        graph = MaximalAncestralGraph
        assert graph.has_inducing_path("C", "D", set())

        graph = AncestralGraph
        graph.latents = "B"
        assert graph.has_inducing_path("C", "D", graph.latents)

        graph = MaximalAncestralGraph
        graph.latents = "B"
        assert graph.has_inducing_path("C", "D", graph.latents)

    def test_check_new_unshielded_collider(self):
        graph = _CoreGraph()

        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "--")

        assert graph._check_new_unshielded_collider(u="C", v="B") is True
        assert graph._check_new_unshielded_collider(u="B", v="C") is False

        graph.add_edge("A", "C", "--")

        assert graph._check_new_unshielded_collider(u="C", v="B") is False

    # def test_get_directed_subgraph(self):
    #     graph = _CoreGraph()

    #     graph.add_edge("A", "B", "->")
    #     graph.add_edge("B", "C", "->")
    #     graph.add_edge("C", "D", "<>")
    #     graph.add_edge("D", "E", "--")
    #     graph.latents = "E"

    #     subgraph = graph.get_directed_subgraph()

    #     assert isinstance(
    #         subgraph, _CoreGraph
    #     )  # TODO(@daehyun99): [#2385] Refactoring _CoreGraph -> DAG when Refactor DAG
    #     assert set(subgraph.nodes()) == {"A", "B", "C", "D", "E"}
    #     assert set(subgraph.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}
    #     assert subgraph.latents == {"E"}
