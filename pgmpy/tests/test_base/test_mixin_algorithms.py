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

        # Creating edges
        ...

    def test_has_cycle(self):
        """
        Testing `_has_cycle` method of Graph class(_CoreGraph, DAG, ADMG)
        """
        ...

    def test_get_ancestral_graph(self):
        """
        Testing `_get_ancestral_graph` method of All graph class
        """
        # _CoreGraph
        graph1 = _CoreGraph()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "--")]
        graph1.add_edges_from(edges)

        new_graph1 = graph1.get_ancestral_graph("C")

        assert f"{type(new_graph1)}" == "<class 'pgmpy.base._base._CoreGraph'>"
        assert set(new_graph1.nodes()) == set(["A", "B", "C"])
        assert set(new_graph1.get_edges(keys=True, data=True)) == set(
            [("A", "B", 0, "->"), ("B", "C", 0, "->")]
        )

        # ADMG
        graph2 = ADMG()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "->")]
        graph2.add_edges_from(edges)

        new_graph2 = graph2.get_ancestral_graph("C")

        assert f"{type(new_graph2)}" == "<class 'pgmpy.base.ADMG.ADMG'>"
        assert set(new_graph2.nodes()) == set(["A", "B", "C"])
        assert set(new_graph2.get_edges(keys=True, data=True)) == set(
            [("A", "B", 0, "->"), ("B", "C", 0, "->")]
        )

        # MAG
        graph3 = MAG()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "--")]
        graph3.add_edges_from(edges)

        new_graph3 = graph3.get_ancestral_graph("C")

        assert f"{type(new_graph3)}" == "<class 'pgmpy.base.MAG.MAG'>"
        assert set(new_graph3.nodes()) == set(["A", "B", "C"])
        assert set(new_graph3.get_edges(keys=True, data=True)) == set(
            [("A", "B", 0, "->"), ("B", "C", 0, "->")]
        )

        # DAG
        ...

        # PAG
        ...

        # UndirectedGraph
        ...
