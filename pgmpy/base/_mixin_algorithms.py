#!/usr/bin/env python3

from typing import Hashable, Set


class _GraphAlgorithmMixin:
    """Mixin class for causal graph's algorithms."""

    def is_collider(self, u: Hashable, v: Hashable, w: Hashable):
        """


        Parameters
        ----------
        u, v, w : node

        Returns
        -------
        bool

        See Also
        --------

        Notes
        -----

        Examples
        --------

        References
        ----------
        [1] Zhang, Jiji. "Causal Reasoning with Ancestral Graphs."
        Journal of Machine Learning Research 9 (2008): 1437-1474.
        """
        ...

    def is_m_separator(self, x: Set, y: Set, z: Set):
        """


        Parameters
        ----------

        Returns
        -------
        bool

        See Also
        --------
        is_m_connected()

        Notes
        -----
        This implementation is based on the 'TestSep' algorithm [1].
        The pseudo-code logic is as follows:

        .. code-block:: text

            function TestSep(G, X, Y, Z)
                P <- { (Wait_Direction, x) | x in X }   # Pending visits
                Q <- P                                  # History (visited)

                while P is not empty do
                    Let (e, T) be a pair in P
                    Remove (e, T) from P

                    for all neighbors N of T do
                        Let T and N be connected by edge f
                        if (e, T, f) is m-connecting given Z and (f, N) not in Q then
                            Add (f, N) to P and Q
                return true

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...

    def is_m_connected(self, x: Set, y: Set, z: Set):
        """


        Parameters
        ----------

        Returns
        -------
        bool

        See Also
        --------
        is_m_separator()

        Notes
        -----

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        return not self.is_m_separator(x, y, z)

    def is_minimal_m_separator(self, x: Set, y: Set, z: Set):
        """


        Parameters
        ----------

        Returns
        -------
        bool

        See Also
        --------

        Notes
        -----
        [TESTMINSEP]

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...

    def get_m_separator(self, x: Set, y: Set, i: Set, r: Set):
        """


        Parameters
        ----------

        Returns
        -------
        M-Separation: set


        See Also
        --------

        Notes
        -----
        [FINDSEP]

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...

    def get_minimal_m_separator(self, x: Set, y: Set, i: Set, r: Set):
        """


        Parameters
        ----------

        Returns
        -------
        M-Separation: set


        See Also
        --------

        Notes
        -----
        [FINDMINSEP]

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...

    def get_m_separators(self, x: Set, y: Set, i: Set, r: Set):
        """


        Parameters
        ----------

        Returns
        -------
        List of M-Separation: list of set

        See Also
        --------

        Notes
        -----
        [LISTSEP]

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...

    def get_minimal_m_separators(self, x: Set, y: Set, i: Set, r: Set):
        """


        Parameters
        ----------

        Returns
        -------
        List of M-Separation: list of set

        See Also
        --------

        Notes
        -----
        [LISTMINSEP]

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...
