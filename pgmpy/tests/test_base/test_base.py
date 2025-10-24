#!/usr/bin/env python3

import unittest

from pgmpy.base import CoreGraph


class TestCoreGraph(unittest.TestCase):
    def setUp(self):
        self.graph = CoreGraph()

    def test_is_directed(self):
        self.assertFalse(self.graph.is_directed())

    def test_is_multigraph(self):
        self.assertTrue(self.graph.is_multigraph())

    def test_add_edge(self):
        """Test the `add_edge` method of the `CoreGraph` class."""
        ...

    def test_add_edges_from(self):
        """Test the `add_edges_from` method of the `CoreGraph` class."""
        ...

    def test_remove_edge(self):
        """Test the `remove_edge` method of the `CoreGraph` class."""
        ...

    def test_remove_edges_from(self):
        """Test the `remove_edges_from` method of the `CoreGraph` class."""
        ...

    def test_equality(self):
        """Test the `__eq__` method of the `CoreGraph` class."""
        ...

    def test_copy(self):
        """Test the `copy` method of the `CoreGraph` class."""
        ...
