#!/usr/bin/env python3

import unittest

import pytest

from pgmpy.base import _CoreGraph


class TestCoreGraph(unittest.TestCase):
    def setUp(self):
        self.graph = _CoreGraph()

    def test_init_empty(self):
        """Test the initialization of an empty `_CoreGraph`"""
        assert len(self.graph.nodes) == 0
        assert len(self.graph.edges) == 0
        assert self.graph.exposures == set()
        assert self.graph.outcomes == set()
        assert self.graph.latents == set()
        assert self.graph.get_role_dict() == {}

    def test_init_with_values(self):
        """Test the initialization of a `_CoreGraph` with values"""
        # Task1: Test the initialization of a `_CoreGraph` with nodes.
        # nodes = ...
        # graph1 = _CoreGraph()

        assert len(self.graph1.nodes) == ...
        assert len(self.graph1.edges) == ...
        assert self.graph1.exposures == ...
        assert self.graph1.outcomes == ...
        assert self.graph1.latents == ...
        assert self.graph1.get_role_dict() == ...

        # Task2: Test the initialization of a `_CoreGraph` with edges.
        # edges = ...
        # graph2 = _CoreGraph()

        assert len(self.graph2.nodes) == ...
        assert len(self.graph2.edges) == ...
        assert self.graph2.exposures == ...
        assert self.graph2.outcomes == ...
        assert self.graph2.latents == ...
        assert self.graph2.get_role_dict() == ...

        # Task3: Test the initialization of a `_CoreGraph` with exposures.
        # exposures = ...
        # graph3 = _CoreGraph()

        assert len(self.graph3.nodes) == ...
        assert len(self.graph3.edges) == ...
        assert self.graph3.exposures == ...
        assert self.graph3.outcomes == ...
        assert self.graph3.latents == ...
        assert self.graph3.get_role_dict() == ...

        # Task4: Test the initialization of a `_CoreGraph` with outcomes.
        # outcomes = ...
        # graph4 = _CoreGraph()

        assert len(self.graph4.nodes) == ...
        assert len(self.graph4.edges) == ...
        assert self.graph4.exposures == ...
        assert self.graph4.outcomes == ...
        assert self.graph4.latents == ...
        assert self.graph4.get_role_dict() == ...

        # Task5: Test the initialization of a `_CoreGraph` with latents.
        # latents = ...
        # graph5 = _CoreGraph()

        assert len(self.graph5.nodes) == ...
        assert len(self.graph5.edges) == ...
        assert self.graph5.exposures == ...
        assert self.graph5.outcomes == ...
        assert self.graph5.latents == ...
        assert self.graph5.get_role_dict() == ...

        # Task6: Test the initialization of a `_CoreGraph` with roles.
        # roles = ...
        # graph6 = _CoreGraph()

        assert len(self.graph6.nodes) == ...
        assert len(self.graph6.edges) == ...
        assert self.graph6.exposures == ...
        assert self.graph6.outcomes == ...
        assert self.graph6.latents == ...
        assert self.graph6.get_role_dict() == ...

        # Task7: Test the initialization of a `_CoreGraph` with values.
        # graph7 = _CoreGraph()

        assert len(self.graph7.nodes) == ...
        assert len(self.graph7.edges) == ...
        assert self.graph7.exposures == ...
        assert self.graph7.outcomes == ...
        assert self.graph7.latents == ...
        assert self.graph7.get_role_dict() == ...

    def test_add_edge(self):
        """Test the `add_edge` method of the `_CoreGraph` class."""
        # Task1: Test adding the direct edge of a `_CoreGraph`.
        graph1 = _CoreGraph()
        graph1.add_edge("A", "C", "->")  # Task1-1: Add edge
        graph1.add_edge("C", "B", "<-")

        assert graph1.has_edge("A", "C") == True  # Task1-2: Test `has_edge()`
        assert graph1.has_edge("C", "B") == True

        # assert graph1["A"]["C"] # Task1-3: Test edgeview

        assert len(graph1.nodes) == 3  # Task1-4: Test `_CoreGraph()`
        assert len(graph1.edges) == 2
        assert graph1.exposures == set()
        assert graph1.outcomes == set()
        assert graph1.latents == set()
        assert graph1.get_role_dict() == {}

        # Task2: Test adding the undirect edge of a `_CoreGraph`.
        graph2 = _CoreGraph()
        graph2.add_edge("A", "C", "--")  # Task2-1: Add edge
        graph2.add_edge("C", "B", "--")

        assert graph2.has_edge("A", "C") == True  # Task2-2: Test `has_edge()`
        assert graph2.has_edge("C", "B") == True

        # assert graph2["A"]["C"] # Task2-3: Test edgeview

        assert len(graph2.nodes) == 3  # Task2-4: Test `_CoreGraph()`
        assert len(graph2.edges) == 2
        assert graph2.exposures == set()
        assert graph2.outcomes == set()
        assert graph2.latents == set()
        assert graph2.get_role_dict() == {}

        # Task3: Test adding the bidirect edge of a `_CoreGraph`.
        graph3 = _CoreGraph()
        graph3.add_edge("A", "C", "<>")  # Task3-1: Add edge
        graph3.add_edge("C", "B", "<>")

        assert graph3.has_edge("A", "C") == True  # Task3-2: Test `has_edge()`
        assert graph3.has_edge("C", "B") == True

        # assert graph3["A"]["C"] # Task3-3: Test edgeview

        assert len(graph3.nodes) == 3  # Task3-4: Test `_CoreGraph()`
        assert len(graph3.edges) == 2
        assert graph3.exposures == set()
        assert graph3.outcomes == set()
        assert graph3.latents == set()
        assert graph3.get_role_dict() == {}

        # Task4: Test adding the unknown edge of a `_CoreGraph`.
        graph4 = _CoreGraph()
        graph4.add_edge("A", "C", "-*")  # Task4-1: Add edge
        graph4.add_edge("C", "B", "*-")
        graph4.add_edge("D", "E", "*>")
        graph4.add_edge("E", "F", "<*")
        graph4.add_edge("G", "H", "**")

        assert graph4.has_edge("A", "C") == True  # Task4-2: Test `has_edge()`
        assert graph4.has_edge("C", "B") == True

        # assert graph4["A"]["C"] # Task4-3: Test edgeview

        assert len(graph4.nodes) == 8  # Task4-4: Test `_CoreGraph()`
        assert len(graph4.edges) == 5
        assert graph4.exposures == set()
        assert graph4.outcomes == set()
        assert graph4.latents == set()
        assert graph4.get_role_dict() == {}

        # Task5: Test adding multiedges of a `_CoreGraph`.
        graph5 = _CoreGraph()
        graph5.add_edge("A", "B", "->")  # Task5-1: Add edge
        graph5.add_edge("A", "B", "<>")
        graph5.add_edge("A", "B", "--")
        graph5.add_edge("A", "B", "**")

        assert graph5.has_edge("A", "B") == True  # Task5-2: Test `has_edge()`

        # assert graph5["A"]["C"] # Task5-3: Test edgeview

        assert len(graph5.nodes) == 2  # Task5-4: Test `_CoreGraph()`
        assert len(graph5.edges) == 4
        assert graph5.exposures == set()
        assert graph5.outcomes == set()
        assert graph5.latents == set()
        assert graph5.get_role_dict() == {}

        # Task6: Test failing add edge of a `_CoreGraph`.
        graph6 = _CoreGraph()

        with pytest.raises(ValueError):
            graph6.add_edge("A", "A", "->")

        with pytest.raises(ValueError):
            graph6.add_edge("A", "B", "-->")

        with pytest.raises(ValueError):
            graph6.add_edge("A", "B", "Invalid_value")

        assert graph6.has_edge("A", "B") == False  # Task6-2: Test `has_edge()`

        # assert graph6["A"]["B"] # Task6-3: Test edgeview

        assert len(graph6.nodes) == 0  # Task6-4: Test `_CoreGraph()`
        assert len(graph6.edges) == 0
        assert graph6.exposures == set()
        assert graph6.outcomes == set()
        assert graph6.latents == set()
        assert graph6.get_role_dict() == {}

    def test_add_edges_from(self):
        """Test the `add_edges_from` method of the `_CoreGraph` class."""
        # Task1: Test adding the direct edges of a `_CoreGraph`.
        ...

        # Task2: Test adding the undirect edges of a `_CoreGraph`.
        ...

        # Task3: Test adding the bidirect edges of a `_CoreGraph`.
        ...

        # Task4: Test adding the unknown edges of a `_CoreGraph`.
        ...

        # Task5: Test adding the various edge of a `_CoreGraph`.
        ...

        # Task6: Test adding multiedges of a `_CoreGraph`.
        ...

        # Task7: Test failing add edges of a `_CoreGraph`.
        ...

    def test_remove_edge(self):
        """Test the `remove_edge` method of the `_CoreGraph` class."""
        # Task1: Test removing the direct edge of a `_CoreGraph`.
        ...

        # Task2: Test removing the undirect edge of a `_CoreGraph`.
        ...

        # Task3: Test removing the bidirect edge of a `_CoreGraph`.
        ...

        # Task4: Test removing the unknown edge of a `_CoreGraph`.
        ...

        # Task5: Test removing multiedges of a `_CoreGraph`.
        ...

        # Task6: Test failing remove edge of a `_CoreGraph`.
        ...

    def test_remove_edges_from(self):
        """Test the `remove_edges_from` method of the `_CoreGraph` class."""
        # Task1: Test removing the direct edges of a `_CoreGraph`.
        ...

        # Task2: Test removing the undirect edges of a `_CoreGraph`.
        ...

        # Task3: Test removing the bidirect edges of a `_CoreGraph`.
        ...

        # Task4: Test removing the unknown edges of a `_CoreGraph`.
        ...

        # Task5: Test removing the various edge of a `_CoreGraph`.
        ...

        # Task6: Test removing multiedges of a `_CoreGraph`.
        ...

        # Task7: Test failing remove edges of a `_CoreGraph`.
        ...

    def test_equality(self):
        """Test the `__eq__` method of the `_CoreGraph` class."""
        # Task1: Test the `__eq__` method of the empty `_CoreGraph` class.
        ...

        # Task2: Test the `__eq__` method of the `_CoreGraph` class with values.
        ...

        # Task3: Test faling the `__eq__` method of the `_CoreGraph` class.
        ...

    def test_copy(self):
        """Test the `copy` method of the `_CoreGraph` class."""
        # Task1: Test the `copy` method of the empty `_CoreGraph` class.
        ...

        # Task2: Test the `copy` method of the `_CoreGraph` class with values.
        ...

        # Task3: Test faling the `copy` method of the `_CoreGraph` class.
        ...
