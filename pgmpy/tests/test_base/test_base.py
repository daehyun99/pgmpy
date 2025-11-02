#!/usr/bin/env python3

import pytest

from pgmpy.base import ADMG, DAG, PDAG, _CoreGraph


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

    def test_init_with_nodes(self):
        """Test the initialization of a `_CoreGraph` with nodes."""
        edges = [("A", "B", "->"), ("B", "C", "->")]
        graph = _CoreGraph(ebunch=edges)

        assert sorted(graph.nodes) == ["A", "B", "C"]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_init_with_edges(self):
        """Test the initialization of a `_CoreGraph` with edges."""
        edges = [("A", "B", "--"), ("A", "B", "-o"), ("B", "C", "<>")]
        graph = _CoreGraph(ebunch=edges)

        assert sorted(graph.edges(data=True), key=lambda x: (x[0], x[1])) == [
            ("A", "B", {"type": "--"}),
            ("A", "B", {"type": "-o"}),
            ("B", "C", {"type": "<>"}),
        ]
        check_graph_status(graph, 3, 3, set(), set(), set(), {})

    def test_init_with_exposures(self):
        """Test the initialization of a `_CoreGraph` with exposures."""
        edges = [("A", "B", "->")]
        graph = _CoreGraph(ebunch=edges, exposures=["A"])

        assert sorted(graph.exposures) == ["A"]
        check_graph_status(graph, 2, 1, {"A"}, set(), set(), {"exposures": ["A"]})

    def test_init_with_outcomes(self):
        """Test the initialization of a `_CoreGraph` with outcomes."""
        edges = [("A", "B", "->")]
        graph = _CoreGraph(ebunch=edges, outcomes=["B"])

        assert sorted(graph.outcomes) == ["B"]
        check_graph_status(graph, 2, 1, set(), {"B"}, set(), {"outcomes": ["B"]})

    def test_init_with_latents(self):
        """Test the initialization of a `_CoreGraph` with latents."""
        edges = [("A", "B", "->")]
        graph = _CoreGraph(ebunch=edges, latents=["A"])

        assert sorted(graph.latents) == ["A"]
        check_graph_status(graph, 2, 1, set(), set(), {"A"}, {"latents": ["A"]})

    def test_init_with_roles(self):
        """Test the initialization of a `_CoreGraph` with roles."""
        edges = [("A", "B", "->")]
        graph = _CoreGraph(ebunch=edges, roles={"test_role": ["A"]})

        assert sorted(graph.get_roles()) == ["test_role"]
        check_graph_status(graph, 2, 1, set(), set(), set(), {"test_role": ["A"]})

    def test_init_with_all_values(self):
        """Test the initialization of a `_CoreGraph` with all values."""
        edges = [("A", "B", "->"), ("B", "C", "oo")]
        graph = _CoreGraph(
            ebunch=edges,
            exposures=["A"],
            outcomes=["B"],
            latents=["C"],
            roles={"test_role": ["A"]},
        )

        check_graph_status(
            graph,
            3,
            2,
            {"A"},
            {"B"},
            {"C"},
            {
                "exposures": ["A"],
                "outcomes": ["B"],
                "latents": ["C"],
                "test_role": ["A"],
            },
        )

    def test_init_fails(self):
        """Test failing the initialization of a `_CoreGraph`."""
        # Task8: Test failing the initialization of a `_CoreGraph` with values.
        graph = _CoreGraph()

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), (None, "A", "->"), ("B", "C", "->")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), ("A", None, "->"), ("B", "C", "->")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # same node error
            edges = [("A", "A", "->")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # same nodes error
            edges = [("A", "B", "->"), ("A", "A", "->"), ("C", "D", "--")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `type` value
            edges = [("A", "B", "-->")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # miss `type` value
            edges = [("A", "B")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `type` values
            edges = [("A", "B", "->"), ("A", "C", "o-->"), ("C", "D", "--")]
            graph = _CoreGraph(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # Granting a role to a node that is not owned.
            roles = {"test_role": "A"}
            graph = _CoreGraph(roles=roles)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # Granting a role to a node that is not owned.
            edges = [("A", "B", "->")]
            roles = {"test_role1": "A", "test_role2": "C", "test_role3": "B"}
            graph = _CoreGraph(ebunch=edges, roles=roles)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_add_directed_edge(self):
        """Test adding the direct edge of a `_CoreGraph`."""
        graph = _CoreGraph()
        graph.add_edge("A", "C", "->")
        graph.add_edge("C", "B", "<-")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("C", "B")

        assert sorted(graph.edges(data=True)) == [
            ("A", "C", {"type": "->"}),
            ("C", "B", {"type": "<-"}),
        ]

        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_add_undirected_edge(self):
        """Test adding the undirect edge of a `_CoreGraph`."""
        graph = _CoreGraph()
        graph.add_edge("A", "C", "--")
        graph.add_edge("C", "B", "--")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("C", "B")

        assert sorted(graph.edges(data=True)) == [
            ("A", "C", {"type": "--"}),
            ("C", "B", {"type": "--"}),
        ]

        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_add_bidirected_edge(self):
        """Test adding the bidirect edge of a `_CoreGraph`."""
        graph = _CoreGraph()
        graph.add_edge("A", "C", "<>")
        graph.add_edge("C", "B", "<>")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("C", "B")

        assert sorted(graph.edges(data=True)) == [
            ("A", "C", {"type": "<>"}),
            ("C", "B", {"type": "<>"}),
        ]

        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_add_unknown_edge(self):
        """Test adding the unknown edge of a `_CoreGraph`."""
        graph = _CoreGraph()
        graph.add_edge("A", "C", "-o")
        graph.add_edge("C", "B", "o-")
        graph.add_edge("D", "E", "o>")
        graph.add_edge("E", "F", "<o")
        graph.add_edge("G", "H", "oo")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("C", "B")

        assert sorted(graph.edges(data=True)) == [
            ("A", "C", {"type": "-o"}),
            ("C", "B", {"type": "o-"}),
            ("D", "E", {"type": "o>"}),
            ("E", "F", {"type": "<o"}),
            ("G", "H", {"type": "oo"}),
        ]

        check_graph_status(graph, 8, 5, set(), set(), set(), {})

    def test_add_multiedges(self):
        """Test adding multiedges of a `_CoreGraph`."""
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        graph.add_edge("A", "B", "--")
        graph.add_edge("A", "B", "oo")

        assert graph.has_edge("A", "B")

        assert sorted(graph.edges(data=True), key=lambda x: (x[0], x[1])) == [
            ("A", "B", {"type": "->"}),
            ("A", "B", {"type": "<>"}),
            ("A", "B", {"type": "--"}),
            ("A", "B", {"type": "oo"}),
        ]

        check_graph_status(graph, 2, 4, set(), set(), set(), {})

    def test_add_edge_fails(self):
        """Test failing add edge of a `_CoreGraph`."""
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.add_edge("A", "A", "->")

        with pytest.raises(ValueError):
            graph.add_edge("A", "B", "-->")

        with pytest.raises(ValueError):
            graph.add_edge("A", "B")

        with pytest.raises(ValueError):
            graph.add_edge("A", "B", "Invalid_value")

        assert not graph.has_edge("A", "B")

        assert sorted(graph.edges(data=True)) == []

        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_add_directed_edges_from(self):
        """Test adding the direct edges of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("B", "C", "->")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(data=True)) == [
            ("A", "B", {"type": "->"}),
            ("B", "C", {"type": "->"}),
        ]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_add_undirected_edges_from(self):
        """Test adding the undirect edges of a `_CoreGraph`."""
        edges = [("A", "B", "--"), ("B", "C", "--")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(data=True)) == [
            ("A", "B", {"type": "--"}),
            ("B", "C", {"type": "--"}),
        ]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_add_bidirected_edges_from(self):
        """Test adding the bidirect edges of a `_CoreGraph`."""
        edges = [("A", "B", "<>"), ("B", "C", "<>")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(data=True)) == [
            ("A", "B", {"type": "<>"}),
            ("B", "C", {"type": "<>"}),
        ]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_add_unknown_edges_from(self):
        """Test adding the unknown edges of a `_CoreGraph`."""
        edges = [("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(data=True)) == [
            ("A", "B", {"type": "-o"}),
            ("B", "C", {"type": "o-"}),
            ("C", "D", {"type": "oo"}),
        ]
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_add_various_edges_from(self):
        """Test adding the various edge of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("B", "C", "--"), ("C", "D", "<>")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(data=True)) == [
            ("A", "B", {"type": "->"}),
            ("B", "C", {"type": "--"}),
            ("C", "D", {"type": "<>"}),
        ]
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

    def test_add_multiedges_from(self):
        """Test adding multiedges of a `_CoreGraph`."""
        edges = [("A", "B", "->"), ("A", "B", "--"), ("A", "B", "oo")]
        graph = _CoreGraph()
        graph.add_edges_from(ebunch=edges)

        assert sorted(graph.edges(data=True), key=lambda x: (x[0], x[1])) == [
            ("A", "B", {"type": "->"}),
            ("A", "B", {"type": "--"}),
            ("A", "B", {"type": "oo"}),
        ]
        check_graph_status(graph, 2, 3, set(), set(), set(), {})

    def test_add_edges_from_fails(self):
        """Test failing add edges of a `_CoreGraph`."""
        graph = _CoreGraph()

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), (None, "A", "->"), ("B", "C", "->")]
            graph.add_edges_from(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), ("A", None, "->"), ("B", "C", "->")]
            graph.add_edges_from(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # miss `type` value
            edges = [("A", "B", "->"), ("A", "C"), ("B", "C", "->")]
            graph.add_edges_from(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # same node error
            edges = [("A", "B", "->"), ("A", "A", "->"), ("B", "C", "->")]
            graph.add_edges_from(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `type` value
            edges = [("A", "B", "->"), ("B", "C", "-->"), ("C", "D", "->")]
            graph.add_edges_from(ebunch=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_remove_edge(self):
        """Test the `remove_edge` method of the `_CoreGraph` class."""
        # Task1: Test removing the direct edge of a `_CoreGraph`.
        edges = [("A", "B", "->"), ("B", "C", "<-")]
        graph1 = _CoreGraph(ebunch=edges)

        graph1.remove_edge("A", "B", "->")  # Task1-1: Remove edge
        graph1.remove_edge("B", "C", "<-")

        assert graph1.has_edge("A", "B") == False  # Task1-2: Test `has_edge()`
        assert graph1.has_edge("B", "C") == False

        assert sorted(graph1.edges(data=True)) == []  # Task1-3: Test edgeview

        check_graph_status(graph1, 3, 0, set(), set(), set(), {})

        # Task2: Test removing the undirect edge of a `_CoreGraph`.
        edges = [("A", "B", "--"), ("B", "C", "--")]
        graph2 = _CoreGraph(ebunch=edges)

        graph2.remove_edge("A", "B", "--")  # Task1-1: Remove edge
        graph2.remove_edge("B", "C", "--")

        assert graph2.has_edge("A", "B") == False  # Task1-2: Test `has_edge()`
        assert graph2.has_edge("B", "C") == False

        assert sorted(graph2.edges(data=True)) == []  # Task1-3: Test edgeview

        check_graph_status(graph2, 3, 0, set(), set(), set(), {})

        # Task3: Test removing the bidirect edge of a `_CoreGraph`.
        edges = [("A", "B", "<>"), ("B", "C", "<>")]
        graph3 = _CoreGraph(ebunch=edges)

        graph3.remove_edge("A", "B", "<>")  # Task1-1: Remove edge
        graph3.remove_edge("B", "C", "<>")

        assert graph3.has_edge("A", "B") == False  # Task1-2: Test `has_edge()`
        assert graph3.has_edge("B", "C") == False

        assert sorted(graph3.edges(data=True)) == []  # Task1-3: Test edgeview

        check_graph_status(graph3, 3, 0, set(), set(), set(), {})

        # Task4: Test removing the unknown edge of a `_CoreGraph`.
        edges = [("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")]
        graph4 = _CoreGraph(ebunch=edges)

        graph4.remove_edge("A", "B", "-o")  # Task1-1: Remove edge
        graph4.remove_edge("B", "C", "o-")
        graph4.remove_edge("C", "D", "oo")

        assert graph4.has_edge("A", "B") == False  # Task1-2: Test `has_edge()`
        assert graph4.has_edge("B", "C") == False
        assert graph4.has_edge("C", "D") == False

        assert sorted(graph4.edges(data=True)) == []  # Task1-3: Test edgeview

        check_graph_status(graph4, 4, 0, set(), set(), set(), {})

        # Task5: Test removing multiedges of a `_CoreGraph`.
        edges = [("A", "B", "->"), ("A", "B", "->"), ("A", "B", "--")]
        graph5 = _CoreGraph(ebunch=edges)

        graph5.remove_edge("A", "B", "->")  # Task1-1: Remove edge
        graph5.remove_edge("A", "B", "->")
        graph5.remove_edge("A", "B", "--")

        assert graph5.has_edge("A", "B") == False  # Task1-2: Test `has_edge()`

        assert sorted(graph5.edges(data=True)) == []  # Task1-3: Test edgeview

        check_graph_status(graph5, 2, 0, set(), set(), set(), {})

        # Task6: Test failing remove edge of a `_CoreGraph`.

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), ("B", "C", "->")]
            graph6 = _CoreGraph(ebunch=edges)
            graph6.remove_edge("A", "B", "->")
            graph6.remove_edge(None, "C", "->")
        check_graph_status(graph6, 3, 1, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), ("B", "C", "->")]
            graph6 = _CoreGraph(ebunch=edges)
            graph6.remove_edge("A", "B", "->")
            graph6.remove_edge("B", None, "->")
        check_graph_status(graph6, 3, 1, set(), set(), set(), {})

        with pytest.raises(ValueError):  # miss `type` value
            edges = [("A", "B", "->"), ("B", "C", "->")]
            graph6 = _CoreGraph(ebunch=edges)
            graph6.remove_edge("A", "B", "->")
            graph6.remove_edge("B", "C")
        check_graph_status(graph6, 3, 1, set(), set(), set(), {})

        with pytest.raises(ValueError):  # same node error
            edges = [("A", "B", "->"), ("B", "C", "->")]
            graph6 = _CoreGraph(ebunch=edges)
            graph6.remove_edge("A", "B", "->")
            graph6.remove_edge("B", "B", "->")
        check_graph_status(graph6, 3, 1, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `type` value
            edges = [("A", "B", "->"), ("B", "C", "->")]
            graph6 = _CoreGraph(ebunch=edges)
            graph6.remove_edge("A", "B", "->")
            graph6.remove_edge("B", "C", "invalid_value")
        check_graph_status(graph6, 3, 1, set(), set(), set(), {})

    def test_remove_edges_from(self):
        """Test the `remove_edges_from` method of the `_CoreGraph` class."""
        # Task1: Test removing the direct edges of a `_CoreGraph`.
        edges = [("A", "B", "->"), ("B", "C", "<-")]
        graph1 = _CoreGraph(ebunch=edges)

        graph1.remove_edges_from(ebunch=edges)  # Task1-1: Remove edge

        assert graph1.has_edge("A", "B") == False  # Task1-2: Test `has_edge()`
        assert graph1.has_edge("B", "C") == False

        assert sorted(graph1.edges(data=True)) == []  # Task1-3: Test edgeview

        check_graph_status(graph1, 3, 0, set(), set(), set(), {})

        # Task2: Test removing the undirect edges of a `_CoreGraph`.
        edges = [("A", "B", "--"), ("B", "C", "--")]
        graph2 = _CoreGraph(ebunch=edges)

        graph2.remove_edges_from(ebunch=edges)  # Task1-1: Remove edge

        assert graph2.has_edge("A", "B") == False  # Task1-2: Test `has_edge()`
        assert graph2.has_edge("B", "C") == False

        assert sorted(graph2.edges(data=True)) == []  # Task1-3: Test edgeview

        check_graph_status(graph2, 3, 0, set(), set(), set(), {})

        # Task3: Test removing the bidirect edges of a `_CoreGraph`.
        edges = [("A", "B", "<>"), ("B", "C", "<>")]
        graph3 = _CoreGraph(ebunch=edges)

        graph3.remove_edges_from(ebunch=edges)  # Task1-1: Remove edge

        assert graph3.has_edge("A", "B") == False  # Task1-2: Test `has_edge()`
        assert graph3.has_edge("B", "C") == False

        assert sorted(graph3.edges(data=True)) == []  # Task1-3: Test edgeview

        check_graph_status(graph3, 3, 0, set(), set(), set(), {})

        # Task4: Test removing the unknown edges of a `_CoreGraph`.
        edges = [("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")]
        graph4 = _CoreGraph(ebunch=edges)

        graph4.remove_edges_from(ebunch=edges)  # Task1-1: Remove edge

        assert graph4.has_edge("A", "B") == False  # Task1-2: Test `has_edge()`
        assert graph4.has_edge("B", "C") == False
        assert graph4.has_edge("C", "D") == False

        assert sorted(graph4.edges(data=True)) == []  # Task1-3: Test edgeview

        check_graph_status(graph4, 4, 0, set(), set(), set(), {})

        # Task5: Test removing the various edge of a `_CoreGraph`.
        edges = [("A", "B", "->"), ("B", "C", "--"), ("C", "D", "<o")]
        graph5 = _CoreGraph(ebunch=edges)

        graph5.remove_edges_from(ebunch=edges)  # Task1-1: Remove edge

        assert graph5.has_edge("A", "B") == False  # Task1-2: Test `has_edge()`
        assert graph5.has_edge("B", "C") == False
        assert graph5.has_edge("C", "D") == False

        assert sorted(graph5.edges(data=True)) == []  # Task1-3: Test edgeview

        check_graph_status(graph5, 4, 0, set(), set(), set(), {})

        # Task6: Test removing multiedges of a `_CoreGraph`.
        edges = [("A", "B", "->"), ("A", "B", "->"), ("A", "B", "--")]
        graph6 = _CoreGraph(ebunch=edges)

        graph6.remove_edges_from(ebunch=edges)  # Task1-1: Remove edge

        assert graph6.has_edge("A", "B") == False  # Task1-2: Test `has_edge()`

        assert sorted(graph6.edges(data=True)) == []  # Task1-3: Test edgeview

        check_graph_status(graph6, 2, 0, set(), set(), set(), {})

        # Task7: Test failing remove edges of a `_CoreGraph`.
        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), ("B", "C", "->")]
            graph7 = _CoreGraph(ebunch=edges)
            graph7.remove_edges_from([("A", "B", "->"), (None, "C", "->")])
        check_graph_status(graph7, 3, 2, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), ("B", "C", "->")]
            graph7 = _CoreGraph(ebunch=edges)
            graph7.remove_edges_from([("A", "B", "->"), ("B", None, "->")])
        check_graph_status(graph7, 3, 2, set(), set(), set(), {})

        with pytest.raises(ValueError):  # miss `type` value
            edges = [("A", "B", "->"), ("B", "C", "->")]
            graph7 = _CoreGraph(ebunch=edges)
            graph7.remove_edges_from([("A", "B", "->"), ("B", "C")])
        check_graph_status(graph7, 3, 2, set(), set(), set(), {})

        with pytest.raises(ValueError):  # same node error
            edges = [("A", "B", "->"), ("B", "C", "->")]
            graph7 = _CoreGraph(ebunch=edges)
            graph7.remove_edges_from([("A", "B", "->"), ("B", "B", "->")])
        check_graph_status(graph7, 3, 2, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `type` value
            edges = [("A", "B", "->"), ("B", "C", "->")]
            graph7 = _CoreGraph(ebunch=edges)
            graph7.remove_edges_from([("A", "B", "->"), ("B", "C", "invalid_value")])
        check_graph_status(graph7, 3, 2, set(), set(), set(), {})

    def test_equality(self):
        """Test the `__eq__` method of the `_CoreGraph` class."""
        # Task1: Test the `__eq__` method of the empty `_CoreGraph` class.
        graph1 = _CoreGraph()
        other1 = _CoreGraph()

        assert graph1.__eq__(other1) == True
        assert other1.__eq__(graph1) == True

        check_graph_status(graph1, 0, 0, set(), set(), set(), {})

        # Task2: Test the `__eq__` method of the `_CoreGraph` class with values.
        # Task2-1: Setting graph and values
        edges = [("A", "B", "->"), ("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]
        roles = {"test_role": ["A", "B"]}
        graph2 = _CoreGraph(
            ebunch=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

        # Task2-2: When the models are the same
        other2 = _CoreGraph()
        other2.add_edges_from(ebunch=edges)
        other2.exposures = exposures
        other2.outcomes = outcomes
        other2.latents = latents
        other2.with_role("test_role", ["A", "B"])

        assert graph2.__eq__(other2) == True
        assert other2.__eq__(graph2) == True

        # Task2-3: When the models differ
        other2_1 = DAG()
        other2_2 = PDAG()
        other2_3 = ADMG()
        other2_4 = "not_graph_class"

        assert graph2.__eq__(other2_1) == False
        assert graph2.__eq__(other2_2) == False
        assert graph2.__eq__(other2_3) == False
        assert graph2.__eq__(other2_4) == False
        assert other2_1.__eq__(graph2) == False
        assert other2_2.__eq__(graph2) == False
        assert other2_3.__eq__(graph2) == False
        assert other2_4.__eq__(graph2) == False

        # Task2-4: When the `ebunch` variables differ between models
        other2_1 = _CoreGraph(
            ebunch=[
                ("A", "B", "->"),
                ("A", "C", "->"),  # differ node
                ("B", "C", "->"),
                ("C", "D", "oo"),
            ],
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )
        other2_2 = _CoreGraph(
            ebunch=[
                ("A", "B", "->"),
                ("A", "B", "--"),  # differ type
                ("B", "C", "->"),
                ("C", "D", "oo"),
            ],
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

        assert graph2.__eq__(other2_1) == False
        assert graph2.__eq__(other2_2) == False
        assert other2_1.__eq__(graph2) == False
        assert other2_2.__eq__(graph2) == False

        # Task2-5: When the `exposures`, `outcomes`, `latents`, and `roles` variables differ between models
        other2_1 = _CoreGraph(
            ebunch=edges,
            exposures=["A", "B"],  # differ exposures
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

        other2_2 = _CoreGraph(
            ebunch=edges,
            exposures=exposures,
            outcomes=["A", "B"],  # differ outcomes
            latents=latents,
            roles=roles,
        )

        other2_3 = _CoreGraph(
            ebunch=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=["A", "B"],  # differ latents
            roles=roles,
        )

        other2_4 = _CoreGraph(
            ebunch=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles={"test_role": ["B", "C"]},  # differ roles
        )

        assert graph2.__eq__(other2_1) == False
        assert graph2.__eq__(other2_2) == False
        assert graph2.__eq__(other2_3) == False
        assert graph2.__eq__(other2_4) == False
        assert other2_1.__eq__(graph2) == False
        assert other2_2.__eq__(graph2) == False
        assert other2_3.__eq__(graph2) == False
        assert other2_4.__eq__(graph2) == False

        check_graph_status(
            graph2,
            4,
            4,
            {"A"},
            {"C"},
            {"D"},
            {
                "exposures": ["A"],
                "outcomes": ["C"],
                "latents": ["D"],
                "test_role": ["A", "B"],
            },
        )

        # Task3: Test failing the `__eq__` method of the `_CoreGraph` class.
        with pytest.raises(ValueError):
            graph3 = _CoreGraph()
            graph3.__eq__()  # empty other
        check_graph_status(graph3, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):
            graph3 = _CoreGraph()
            other3_1 = _CoreGraph()
            other3_2 = _CoreGraph()
            graph3.__eq__(other3_1, other3_2)  # unexpected values
        check_graph_status(graph3, 0, 0, set(), set(), set(), {})

    def test_copy(self):
        """Test the `copy` method of the `_CoreGraph` class."""
        # Task1: Test the `copy` method of the empty `_CoreGraph` class.
        graph1_1 = _CoreGraph()
        graph1_2 = graph1_1.copy()

        assert graph1_1.__eq__(graph1_2) == True
        assert graph1_2.__eq__(graph1_1) == True

        check_graph_status(graph1_1, 0, 0, set(), set(), set(), {})

        # Task2: Test the `copy` method of the `_CoreGraph` class with values.
        # Task2-1: Setting values
        edges = [("A", "B", "->"), ("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]
        roles = {"test_role": ["A", "B"]}

        # Task2-2: only have `ebunch`
        graph2_1 = _CoreGraph(ebunch=edges)
        graph2_2 = graph2_1.copy()

        assert graph2_1.__eq__(graph2_2) == True
        assert graph2_2.__eq__(graph2_1) == True

        check_graph_status(graph2_1, 4, 4, set(), set(), set(), {})

        # Task2-3: have `exposures`, `outcomes`, `latents`
        graph2_1 = _CoreGraph(
            ebunch=edges, exposures=exposures, outcomes=outcomes, latents=latents
        )
        graph2_2 = graph2_1.copy()

        assert graph2_1.__eq__(graph2_2) == True
        assert graph2_2.__eq__(graph2_1) == True

        check_graph_status(
            graph2_1,
            4,
            4,
            {"A"},
            {"C"},
            {"D"},
            {
                "exposures": ["A"],
                "outcomes": ["C"],
                "latents": ["D"],
            },
        )

        # Task2-4: have `roles`
        graph2_1 = _CoreGraph(ebunch=edges, roles=roles)
        graph2_2 = graph2_1.copy()

        assert graph2_1.__eq__(graph2_2) == True
        assert graph2_2.__eq__(graph2_1) == True

        check_graph_status(
            graph2_1,
            4,
            4,
            set(),
            set(),
            set(),
            {
                "test_role": ["A", "B"],
            },
        )

        # Task2-5: has all values
        graph2_1 = _CoreGraph(
            ebunch=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )
        graph2_2 = graph2_1.copy()

        assert graph2_1.__eq__(graph2_2) == True
        assert graph2_2.__eq__(graph2_1) == True

        check_graph_status(
            graph2_1,
            4,
            4,
            {"A"},
            {"C"},
            {"D"},
            {
                "exposures": ["A"],
                "outcomes": ["C"],
                "latents": ["D"],
                "test_role": ["A", "B"],
            },
        )

        # Task3: Test failing the `copy` method of the `_CoreGraph` class.
        with pytest.raises(TypeError):
            graph3_1 = _CoreGraph()
            graph3_2 = graph3_1.copy("invalid_value")

            assert graph3_1.__eq__(graph3_2) == False
        check_graph_status(graph3_1, 0, 0, set(), set(), set(), {})
