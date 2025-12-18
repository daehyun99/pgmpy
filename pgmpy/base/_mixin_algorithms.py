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
        ...

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

        Examples
        --------

        References
        ----------
        [1] zander, Benito van der, Maciej Liskiewicz, and Johannes C. Textor.
        "Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework."
        Artificial Intelligence 270 (2019): 1-40.
        """
        ...
