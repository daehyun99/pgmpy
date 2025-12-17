# import pytest

from pgmpy.base._base import _CoreGraph


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

    def test_get_m_separator(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 1.
        """
        ...

    def test_get_m_separator_with_latent(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 1.
        """
        ...

    def test_get_m_separators(self):
        """
        References
        ----------
        [1] Zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40. Figure 1.
        """
        ...
