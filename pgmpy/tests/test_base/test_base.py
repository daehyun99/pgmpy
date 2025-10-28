#!/usr/bin/env python3

import pytest

from pgmpy.base import _CoreGraph


def check_graph_status(
    graph,
    node_count: int,
    edge_count: int,
    exposures: set,
    outcomes: set,
    latents: set,
    roles: dict,
):
    """Common graph state checking function."""
    assert len(graph.nodes) == node_count
    assert len(graph.edges) == edge_count
    assert graph.exposures == exposures
    assert graph.outcomes == outcomes
    assert graph.latents == latents
    assert graph.get_role_dict() == roles


class TestCoreGraph:
    def test_init_empty(self):
        """Test the initialization of an empty `_CoreGraph`"""
        graph = _CoreGraph()
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_init_with_values(self):
        """Test the initialization of a `_CoreGraph` with values"""
        # Task1: Test the initialization of a `_CoreGraph` with nodes.
        edges = [("A", "B", "->"), ("B", "C", "->")]
        graph1 = _CoreGraph(ebunch=edges)

        assert sorted(graph1.nodes) == ["A", "B", "C"]
        check_graph_status(graph1, 3, 2, set(), set(), set(), {})

        # Task2: Test the initialization of a `_CoreGraph` with edges.
        edges = [("A", "B", "--"), ("A", "B", "-o"), ("B", "C", "<>")]
        graph2 = _CoreGraph(ebunch=edges)

        assert sorted(graph2.edges(data=True)) == [
            ("A", "B", {"type": "--"}),
            ("A", "B", {"type": "-o"}),
            ("B", "C", {"type": "<>"}),
        ]
        check_graph_status(graph2, 3, 3, set(), set(), set(), {})

        # Task3: Test the initialization of a `_CoreGraph` with exposures.
        edges = [("A", "B", "->")]
        graph3 = _CoreGraph(ebunch=edges, exposures=["A"])

        assert sorted(graph3.exposures) == ["A"]
        check_graph_status(graph3, 2, 1, ["A"], set(), set(), {"exposures": ["A"]})

        # Task4: Test the initialization of a `_CoreGraph` with outcomes.
        edges = [("A", "B", "->")]
        graph4 = _CoreGraph(ebunch=edges, outcomes=["B"])

        assert sorted(graph4.outcomes) == ["B"]
        check_graph_status(graph4, 2, 1, set(), ["B"], set(), {"outcomes": ["B"]})

        # Task5: Test the initialization of a `_CoreGraph` with latents.
        edges = [("A", "B", "->")]
        graph5 = _CoreGraph(ebunch=edges, latents=["A"])

        assert sorted(graph5.latents) == ["A"]
        check_graph_status(graph5, 2, 1, set(), set(), ["A"], {"latents": ["A"]})

        # Task6: Test the initialization of a `_CoreGraph` with roles.
        edges = [("A", "B", "->")]
        graph6 = _CoreGraph(ebunch=edges, roles={"test_role": ["A"]})

        assert sorted(graph6.get_roles()) == ["test_role"]
        check_graph_status(graph6, 2, 1, set(), set(), set(), {"test_role": ["A"]})

        # Task7: Test the initialization of a `_CoreGraph` with values.
        edges = [("A", "B", "->"), ("B", "C", "oo")]
        graph7 = _CoreGraph(
            ebunch=edges,
            exposures=["A"],
            outcomes=["B"],
            latents=["C"],
            roles={"test_role": ["A"]},
        )

        check_graph_status(
            graph7,
            3,
            2,
            ["A"],
            ["B"],
            ["C"],
            {
                "exposures": ["A"],
                "outcomes": ["B"],
                "latents": ["C"],
                "test_role": ["A"],
            },
        )

        # Task8: Test failing the initialization of a `_CoreGraph` with values.
        with pytest.raises(ValueError):  # same node error
            edges = [("A", "A", "->")]
            graph8 = _CoreGraph(ebunch=edges)

        with pytest.raises(ValueError):  # same nodes error
            edges = [("A", "B", "->"), ("A", "A", "->"), ("C", "D", "--")]
            graph8 = _CoreGraph(ebunch=edges)

        with pytest.raises(ValueError):  # invalid `type` value
            edges = [("A", "B", "-->")]
            graph8 = _CoreGraph(ebunch=edges)

        with pytest.raises(ValueError):  # invalid `type` values
            edges = [("A", "B", "->"), ("A", "C", "o-->"), ("C", "D", "--")]
            graph8 = _CoreGraph(ebunch=edges)

        with pytest.raises(ValueError):  # Granting a role to a node that is not owned.
            roles = {"test_role": "A"}
            graph8 = _CoreGraph(roles=roles)

        with pytest.raises(ValueError):  # Granting a role to a node that is not owned.
            edges = [("A", "B", "->")]
            roles = {"test_role1": "A", "test_role2": "C", "test_role3": "B"}
            graph8 = _CoreGraph(ebunch=edges, roles=roles)

        check_graph_status(graph8, 0, 0, set(), set(), set(), {})

    def test_add_edge(self):
        """Test the `add_edge` method of the `_CoreGraph` class."""
        # Task1: Test adding the direct edge of a `_CoreGraph`.
        graph1 = _CoreGraph()
        graph1.add_edge("A", "C", "->")  # Task1-1: Add edge
        graph1.add_edge("C", "B", "<-")

        assert graph1.has_edge("A", "C") == True  # Task1-2: Test `has_edge()`
        assert graph1.has_edge("C", "B") == True

        assert sorted(graph1.edges(data=True)) == [  # Task1-3: Test edgeview
            ("A", "C", {"type": "->"}),
            ("C", "B", {"type": "<-"}),
        ]

        check_graph_status(graph1, 3, 2, set(), set(), set(), {})

        # Task2: Test adding the undirect edge of a `_CoreGraph`.
        graph2 = _CoreGraph()
        graph2.add_edge("A", "C", "--")  # Task2-1: Add edge
        graph2.add_edge("C", "B", "--")

        assert graph2.has_edge("A", "C") == True  # Task2-2: Test `has_edge()`
        assert graph2.has_edge("C", "B") == True

        assert sorted(graph2.edges(data=True)) == [  # Task2-3: Test edgeview
            ("A", "C", {"type": "--"}),
            ("C", "B", {"type": "--"}),
        ]

        check_graph_status(graph2, 3, 2, set(), set(), set(), {})

        # Task3: Test adding the bidirect edge of a `_CoreGraph`.
        graph3 = _CoreGraph()
        graph3.add_edge("A", "C", "<>")  # Task3-1: Add edge
        graph3.add_edge("C", "B", "<>")

        assert graph3.has_edge("A", "C") == True  # Task3-2: Test `has_edge()`
        assert graph3.has_edge("C", "B") == True

        assert sorted(graph3.edges(data=True)) == [  # Task3-3: Test edgeview
            ("A", "C", {"type": "<>"}),
            ("C", "B", {"type": "<>"}),
        ]

        check_graph_status(graph3, 3, 2, set(), set(), set(), {})

        # Task4: Test adding the unknown edge of a `_CoreGraph`.
        graph4 = _CoreGraph()
        graph4.add_edge("A", "C", "-o")  # Task4-1: Add edge
        graph4.add_edge("C", "B", "o-")
        graph4.add_edge("D", "E", "o>")
        graph4.add_edge("E", "F", "<o")
        graph4.add_edge("G", "H", "oo")

        assert graph4.has_edge("A", "C") == True  # Task4-2: Test `has_edge()`
        assert graph4.has_edge("C", "B") == True

        assert sorted(graph4.edges(data=True)) == [  # Task4-3: Test edgeview
            ("A", "C", {"type": "-o"}),
            ("C", "B", {"type": "o-"}),
            ("D", "E", {"type": "o>"}),
            ("E", "F", {"type": "<o"}),
            ("G", "H", {"type": "oo"}),
        ]

        check_graph_status(graph4, 8, 5, set(), set(), set(), {})

        # Task5: Test adding multiedges of a `_CoreGraph`.
        graph5 = _CoreGraph()
        graph5.add_edge("A", "B", "->")  # Task5-1: Add edge
        graph5.add_edge("A", "B", "<>")
        graph5.add_edge("A", "B", "--")
        graph5.add_edge("A", "B", "oo")

        assert graph5.has_edge("A", "B") == True  # Task5-2: Test `has_edge()`

        assert sorted(  # Task5-3: Test edgeview
            graph5.edges(data=True), key=lambda x: (x[0], x[1])
        ) == [
            ("A", "B", {"type": "->"}),
            ("A", "B", {"type": "<>"}),
            ("A", "B", {"type": "--"}),
            ("A", "B", {"type": "oo"}),
        ]

        check_graph_status(graph5, 2, 4, set(), set(), set(), {})

        # Task6: Test failing add edge of a `_CoreGraph`.
        graph6 = _CoreGraph()

        with pytest.raises(ValueError):
            graph6.add_edge("A", "A", "->")

        with pytest.raises(ValueError):
            graph6.add_edge("A", "B", "-->")

        with pytest.raises(ValueError):
            graph6.add_edge("A", "B", "Invalid_value")

        assert graph6.has_edge("A", "B") == False  # Task6-2: Test `has_edge()`

        assert sorted(graph6.edges(data=True)) == []  # Task6-3: Test edgeview

        check_graph_status(graph6, 0, 0, set(), set(), set(), {})

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
