import networkx as nx
import pytest

from pgmpy.base import ADMG, MAG
from pgmpy.base._base import _CoreGraph


class TestGraphAlgorithmMixin:
    @pytest.mark.skip(
        reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class."
    )
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

    @pytest.mark.skip(
        reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class."
    )
    def test_is_m_separator(self):
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

        assert graph.is_m_separator("T", "O", "M") == True
        assert graph.is_m_separator("T", "I", "M") == False
        assert graph.is_m_separator("T", "B", "M") == False
        assert graph.is_m_separator("T", "U", "M") == True

        graph = _CoreGraph()
        graph.add_edge("T", "M", "<-")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_m_separator("T", "O", "M") == True
        assert graph.is_m_separator("T", "I", "M") == True
        assert graph.is_m_separator("T", "B", "M") == True
        assert graph.is_m_separator("T", "U", "M") == True

        graph = _CoreGraph()
        graph.add_edge("T", "M", "<>")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_m_separator("T", "O", "M") == True
        assert graph.is_m_separator("T", "I", "M") == False
        assert graph.is_m_separator("T", "B", "M") == False
        assert graph.is_m_separator("T", "U", "M") == True

        graph = _CoreGraph()
        graph.add_edge("T", "M", "--")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_m_separator("T", "O", "M") == True
        assert graph.is_m_separator("T", "I", "M") == True
        assert graph.is_m_separator("T", "B", "M") == True
        assert graph.is_m_separator("T", "U", "M") == True

    @pytest.mark.skip(
        reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class."
    )
    def test_is_m_separator_with_latent(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 3.
        """
        graph = _CoreGraph()
        graph.add_edge("T", "M", "->")
        graph.latents = {"M"}

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_m_separator("T", "O", "M") == False
        assert graph.is_m_separator("T", "I", "M") == True
        assert graph.is_m_separator("T", "B", "M") == True
        assert graph.is_m_separator("T", "U", "M") == False

        graph = _CoreGraph()
        graph.add_edge("T", "M", "<-")
        graph.latents = {"M"}

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_m_separator("T", "O", "M") == False
        assert graph.is_m_separator("T", "I", "M") == False
        assert graph.is_m_separator("T", "B", "M") == False
        assert graph.is_m_separator("T", "U", "M") == False

        graph = _CoreGraph()
        graph.add_edge("T", "M", "<>")
        graph.latents = {"M"}

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_m_separator("T", "O", "M") == False
        assert graph.is_m_separator("T", "I", "M") == True
        assert graph.is_m_separator("T", "B", "M") == True
        assert graph.is_m_separator("T", "U", "M") == False

        graph = _CoreGraph()
        graph.add_edge("T", "M", "--")
        graph.latents = {"M"}

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_m_separator("T", "O", "M") == False
        assert graph.is_m_separator("T", "I", "M") == False
        assert graph.is_m_separator("T", "B", "M") == False
        assert graph.is_m_separator("T", "U", "M") == False

    @pytest.mark.skip(
        reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class."
    )
    def test_is_m_connected(self):
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

        assert graph.is_m_connected("T", "O", "M") == False
        assert graph.is_m_connected("T", "I", "M") == True
        assert graph.is_m_connected("T", "B", "M") == True
        assert graph.is_m_connected("T", "U", "M") == False

        graph = _CoreGraph()
        graph.add_edge("T", "M", "<-")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_m_connected("T", "O", "M") == False
        assert graph.is_m_connected("T", "I", "M") == False
        assert graph.is_m_connected("T", "B", "M") == False
        assert graph.is_m_connected("T", "U", "M") == False

        graph = _CoreGraph()
        graph.add_edge("T", "M", "<>")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_m_connected("T", "O", "M") == False
        assert graph.is_m_connected("T", "I", "M") == True
        assert graph.is_m_connected("T", "B", "M") == True
        assert graph.is_m_connected("T", "U", "M") == False

        graph = _CoreGraph()
        graph.add_edge("T", "M", "--")

        graph.add_edge("M", "O", "->")
        graph.add_edge("M", "I", "<-")
        graph.add_edge("M", "B", "<>")
        graph.add_edge("M", "U", "--")

        assert graph.is_m_connected("T", "O", "M") == False
        assert graph.is_m_connected("T", "I", "M") == False
        assert graph.is_m_connected("T", "B", "M") == False
        assert graph.is_m_connected("T", "U", "M") == False

    @pytest.mark.skip(
        reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class."
    )
    def test_is_minimal_m_separator(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 1.
        """
        # TODO(@daehyun99): [#2384] Implement code logic and test code
        ...

    @pytest.mark.skip(
        reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class."
    )
    def test_get_m_separator(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 1.
        """
        # TODO(@daehyun99): [#2384] Implement code logic and test code
        ...

    @pytest.mark.skip(
        reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class."
    )
    def test_get_m_separator_with_latent(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 1.
        """
        # TODO(@daehyun99): [#2384] Implement code logic and test code
        ...

    @pytest.mark.skip(
        reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class."
    )
    def test_get_minimal_m_separator(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 1.
        """
        # TODO(@daehyun99): [#2384] Implement code logic and test code
        ...

    @pytest.mark.skip(
        reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class."
    )
    def test_get_m_separators(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 1.
        """
        # TODO(@daehyun99): [#2384] Implement code logic and test code
        ...

    @pytest.mark.skip(
        reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class."
    )
    def test_get_m_separators_with_latent(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 1.
        """
        # TODO(@daehyun99): [#2384] Implement code logic and test code
        ...

    @pytest.mark.skip(
        reason="Refactoring: Skip now, because focusing on refactoring ADMG, MAG class."
    )
    def test_get_minimal_m_separators(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 1.

        [2] Takata, Ken.
        "Space-optimal, backtracking algorithms to list the minimal vertex separators of a graph."
        Discrete Applied Mathematics 158 (2010): 1660-1667. Figure 1.
        """
        graph = _CoreGraph()
        graph.add_nodes_from(["A", "two", "three", "four", "five", "B"])

        # TODO(@daehyun99): [#2384] Implement code logic and test code
        ...

    def test_has_cycle(self):
        """
        Testing `_has_cycle` method of Graph class(_CoreGraph, DAG, ADMG)
        """
        # TODO(@daehyun99): [#2385] Consider implement `_has_cycle` method.

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
        assert set(new_graph1.nodes()) == set(["A", "B", "C"])
        assert set(new_graph1.get_edges(keys=True, data=True)) == set(
            [("A", "B", 0, "->"), ("B", "C", 0, "->")]
        )

        # ADMG
        graph2 = ADMG()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "->")]
        graph2.add_edges_from(edges)

        new_graph2 = graph2.get_ancestral_graph("C")

        assert isinstance(new_graph2, ADMG)
        assert set(new_graph2.nodes()) == set(["A", "B", "C"])
        assert set(new_graph2.get_edges(keys=True, data=True)) == set(
            [("A", "B", 0, "->"), ("B", "C", 0, "->")]
        )

        # MAG
        graph3 = MAG()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "--")]
        graph3.add_edges_from(edges)

        new_graph3 = graph3.get_ancestral_graph("C")

        assert isinstance(new_graph3, MAG)
        assert set(new_graph3.nodes()) == set(["A", "B", "C"])
        assert set(new_graph3.get_edges(keys=True, data=True)) == set(
            [("A", "B", 0, "->"), ("B", "C", 0, "->")]
        )

        # TODO(@daehyun99): [#2384] Expand DAG
        ...

        # TODO(@daehyun99): [#2384] Expand PAG
        ...

        # TODO(@daehyun99): [#2384] Expand UndirectedGraph
        ...

    def test_get_markov_blanket(self):
        """"""
        # TODO(@daehyun99): [#2385] Implement code logic and test code
        ...

    def test_has_inducing_path(self):
        """"""
        # TODO(@daehyun99): [#2385] Implement code logic and test code
        ...

    def test_has_direct_path_basic(self):
        """Test code for `direct_path` method"""
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "->")

        assert nx.has_path(graph, "A", "C") is True
        assert graph.has_direct_path("A", "C") is True

    def test_has_direct_path_reverse(self):
        """Test code for `direct_path` method"""
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "<-")
        graph.add_edge("C", "D", "->")

        assert nx.has_path(graph, "A", "D") is True
        assert graph.has_direct_path("A", "D") is False

    def test_has_direct_path_various_edges(self):
        """Test code for `direct_path` method"""
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "<>")

        assert nx.has_path(graph, "A", "C") is True
        assert graph.has_direct_path("A", "C") is False

    def test_has_direct_path_two_path(self):
        """Test code for `direct_path` method"""
        graph = _CoreGraph()

        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "C", "->")
        graph.add_edge("B", "D", "<-")
        graph.add_edge("C", "D", "->")

        assert nx.has_path(graph, "A", "D") is True
        assert graph.has_direct_path("A", "D") is True
