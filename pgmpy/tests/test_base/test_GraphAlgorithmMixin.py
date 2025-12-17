# import pytest

from pgmpy.base._base import _CoreGraph


class TestGraphAlgorithmMixin:
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

    def test_is_m_separaotr_with_latent(self):
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
