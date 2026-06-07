#!/usr/bin/env python3

import pytest

from pgmpy.base import ADMG, DAG, PDAG
from pgmpy.base._base import _CoreGraph


def sample_graph1(edge_type=None):
    """
    Sample graph for testing node searching(`get_*`) method of `_CoreGraph` class.
    Tests node searching methods centered on node `C`.

    Notes
    -----
        +---+             +---+             +---+             +---+             +---+
        | A | [edge_type] | B | [edge_type] | C | [edge_type] | D | [edge_type] | E |
        +---+             +---+             +---+             +---+             +---+
    """
    edges = [
        ("A", "B", edge_type),
        ("B", "C", edge_type),
        ("C", "D", edge_type),
        ("D", "E", edge_type),
    ]
    return _CoreGraph(edge_list=edges)


def sample_graph2(edge_type=None):
    """
    Sample graph for testing node searching(`get_*`) method of `_CoreGraph` class.
    Tests node searching methods centered on node `B`.

    Notes
    -----
                                            +---+
                                [edge_type] | C |
        +---+             +---+             +---+
        | A | [edge_type] | B |
        +---+             +---+             +---+
                                [edge_type] | D |
                                            +---+


    """
    edges = [
        ("A", "B", edge_type),
        ("B", "C", edge_type),
        ("B", "D", edge_type),
    ]
    return _CoreGraph(edge_list=edges)


def sample_graph3():
    """
    sample graph for testing node searching(`get_*`) method of `_CoreGraph` class.

    Notes
    -----
    Used `base_graph` from test_AncestralBase.py.
    Expected to be same as tests in test_AncestralBase.py.
    Used in `test_get_*_with_multiedges` method of `_CoreGraph` class.
    """
    edges = [
        ("A", "B", "->"),
        ("A", "C", "<-"),
        ("A", "D", "oo"),
        ("A", "E", "<>"),
        ("A", "F", "--"),
        ("A", "G", "-o"),
        ("A", "H", "o-"),
        ("A", "I", "o>"),
        ("A", "J", "<o"),
        ("B", "X", "->"),
        ("C", "Y", "<-"),
    ]
    return _CoreGraph(edge_list=edges)


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
    assert len(graph.edges(keys=True, data=True)) == edge_count
    assert graph.exposures == exposures
    assert graph.outcomes == outcomes
    assert graph.latents == latents
    assert graph.get_role_dict() == roles


class TestCoreGraph:
    def test_init(self):
        """Test the initialization of a `_CoreGraph`."""
        # empty graph
        graph = _CoreGraph()
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # nodes from edges
        edges = [("A", "B", "->"), ("B", "C", "->")]
        graph = _CoreGraph(edge_list=edges)

        assert sorted(graph.nodes) == ["A", "B", "C"]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # edges with different edge types
        edges = [("A", "B", "--"), ("A", "B", "-o"), ("B", "C", "<>")]
        graph = _CoreGraph(edge_list=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": "-"}),
            ("A", "B", 1, {"A": "-", "B": "o"}),
            ("B", "C", 0, {"B": ">", "C": ">"}),
        ]
        check_graph_status(graph, 3, 3, set(), set(), set(), {})

        # exposures
        edges = [("A", "B", "->")]
        graph = _CoreGraph(edge_list=edges, exposures=["A"])

        assert sorted(graph.exposures) == ["A"]
        check_graph_status(graph, 2, 1, {"A"}, set(), set(), {"exposures": ["A"]})

        # outcomes
        edges = [("A", "B", "->")]
        graph = _CoreGraph(edge_list=edges, outcomes=["B"])

        assert sorted(graph.outcomes) == ["B"]
        check_graph_status(graph, 2, 1, set(), {"B"}, set(), {"outcomes": ["B"]})

        # latents
        edges = [("A", "B", "->")]
        graph = _CoreGraph(edge_list=edges, latents=["A"])

        assert sorted(graph.latents) == ["A"]
        check_graph_status(graph, 2, 1, set(), set(), {"A"}, {"latents": ["A"]})

        # custom roles
        edges = [("A", "B", "->")]
        graph = _CoreGraph(edge_list=edges, roles={"test_role": ["A"]})

        assert sorted(graph.get_roles()) == ["test_role"]
        check_graph_status(graph, 2, 1, set(), set(), set(), {"test_role": ["A"]})

        # all values together
        edges = [("A", "B", "->"), ("B", "C", "oo"), ("C", "D", "--")]
        graph = _CoreGraph(
            edge_list=edges,
            exposures=["A"],
            outcomes=["B"],
            latents=["C"],
            roles={"test_role": ["D"]},
        )

        check_graph_status(
            graph,
            4,
            3,
            {"A"},
            {"B"},
            {"C"},
            {
                "exposures": ["A"],
                "outcomes": ["B"],
                "latents": ["C"],
                "test_role": ["D"],
            },
        )

        # fails: invalid u/v, same node, unsupported edge type, bad tuple length, unowned role
        graph = _CoreGraph()

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), (None, "A", "->"), ("B", "C", "->")]
            graph = _CoreGraph(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), ("A", None, "->"), ("B", "C", "->")]
            graph = _CoreGraph(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # same node error
            edges = [("A", "A", "->")]
            graph = _CoreGraph(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # same nodes error
            edges = [("A", "B", "->"), ("A", "A", "->"), ("C", "D", "--")]
            graph = _CoreGraph(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `edge_type` value
            edges = [("A", "B", "-->")]
            graph = _CoreGraph(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `edge_type` values
            edges = [("A", "B", "->"), ("A", "C", "o-->"), ("C", "D", "--")]
            graph = _CoreGraph(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError, match="3 elements"):  # 2-tuple edge (missing `edge_type`)
            edges = [("A", "B")]
            graph = _CoreGraph(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError, match="3 elements"):  # 4-tuple edge
            edges = [("A", "B", "key", "->")]
            graph = _CoreGraph(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # Granting a role to a node that is not owned.
            roles = {"test_role": "A"}
            graph = _CoreGraph(roles=roles)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # Granting a role to a node that is not owned.
            edges = [("A", "B", "->")]
            roles = {"test_role1": "A", "test_role2": "C", "test_role3": "B"}
            graph = _CoreGraph(edge_list=edges, roles=roles)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_add_edge(self):
        """Test the `_CoreGraph.add_edge` method."""
        # directed edge
        graph = _CoreGraph()
        graph.add_edge("A", "C", "->")
        graph.add_edge("C", "B", "<-")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "C", 0, {"A": "-", "C": ">"}),
            ("C", "B", 0, {"B": "-", "C": ">"}),
        ]

        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # undirected edge
        graph = _CoreGraph()
        graph.add_edge("A", "C", "--")
        graph.add_edge("C", "B", "--")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("C", "B")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "C", 0, {"A": "-", "C": "-"}),
            ("C", "B", 0, {"B": "-", "C": "-"}),
        ]

        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # bidirected edge
        graph = _CoreGraph()
        graph.add_edge("A", "C", "<>")
        graph.add_edge("C", "B", "<>")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("C", "B")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "C", 0, {"A": ">", "C": ">"}),
            ("C", "B", 0, {"B": ">", "C": ">"}),
        ]

        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # unknown (circle) edges
        graph = _CoreGraph()
        graph.add_edge("A", "C", "-o")
        graph.add_edge("C", "B", "o-")
        graph.add_edge("D", "E", "o>")
        graph.add_edge("E", "F", "<o")
        graph.add_edge("G", "H", "oo")

        assert graph.has_edge("A", "C")
        assert graph.has_edge("C", "B")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "C", 0, {"A": "-", "C": "o"}),
            ("C", "B", 0, {"B": "-", "C": "o"}),
            ("D", "E", 0, {"D": "o", "E": ">"}),
            ("E", "F", 0, {"F": "o", "E": ">"}),
            ("G", "H", 0, {"G": "o", "H": "o"}),
        ]

        check_graph_status(graph, 8, 5, set(), set(), set(), {})

        # multiedges of different types between the same pair
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        graph.add_edge("A", "B", "--")
        graph.add_edge("A", "B", "oo")

        assert graph.has_edge("A", "B")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": ">"}),
            ("A", "B", 1, {"A": ">", "B": ">"}),
            ("A", "B", 2, {"A": "-", "B": "-"}),
            ("A", "B", 3, {"A": "o", "B": "o"}),
        ]

        check_graph_status(graph, 2, 4, set(), set(), set(), {})

        # edge with kwargs
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->", weight=5)
        graph.add_edge("B", "C", "->", weight=8)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"weight": 5, "A": "-", "B": ">"}),
            ("B", "C", 0, {"weight": 8, "B": "-", "C": ">"}),
        ]

        # fails: duplicate edge of the same type is rejected
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        graph.add_edge("A", "B", "--")

        with pytest.raises(ValueError, match="already exists"):
            graph.add_edge("A", "B", "->")

        # fails: directed edge closing a cycle is rejected (`->` and `<-`)
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->")])

        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge("C", "A", "->")

        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge("A", "C", "<-")

        # fails: same node / unsupported edge_type / missing / None
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.add_edge("A", "A", "->")

        with pytest.raises(ValueError):
            graph.add_edge("A", "B", "-->")

        with pytest.raises(ValueError):
            graph.add_edge("A", "B", "Invalid_value")

        with pytest.raises(ValueError):
            graph.add_edge("A", "B", 1)

        with pytest.raises(ValueError):
            graph.add_edge("A", "B", set())

        with pytest.raises(ValueError):
            graph.add_edge("A", "B", dict())

        with pytest.raises(TypeError):  # edge_type is required (no default)
            graph.add_edge("A", "B")

        with pytest.raises(ValueError, match=r"Got \('A', 'B', None\)"):  # None is not a valid edge_type
            graph.add_edge("A", "B", None)

        assert not graph.has_edge("A", "B")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_add_edges_from(self):
        """Test the `_CoreGraph.add_edges_from` method."""
        # directed edges
        edges = [("A", "B", "->"), ("B", "C", "->")]
        graph = _CoreGraph()
        graph.add_edges_from(edge_list=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": ">"}),
            ("B", "C", 0, {"B": "-", "C": ">"}),
        ]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # undirected edges
        edges = [("A", "B", "--"), ("B", "C", "--")]
        graph = _CoreGraph()
        graph.add_edges_from(edge_list=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": "-"}),
            ("B", "C", 0, {"B": "-", "C": "-"}),
        ]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # bidirected edges
        edges = [("A", "B", "<>"), ("B", "C", "<>")]
        graph = _CoreGraph()
        graph.add_edges_from(edge_list=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": ">", "B": ">"}),
            ("B", "C", 0, {"B": ">", "C": ">"}),
        ]
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # unknown (circle) edges
        edges = [("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")]
        graph = _CoreGraph()
        graph.add_edges_from(edge_list=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": "o"}),
            ("B", "C", 0, {"B": "o", "C": "-"}),
            ("C", "D", 0, {"C": "o", "D": "o"}),
        ]
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # various edge types
        edges = [("A", "B", "->"), ("B", "C", "--"), ("C", "D", "<>")]
        graph = _CoreGraph()
        graph.add_edges_from(edge_list=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": ">"}),
            ("B", "C", 0, {"B": "-", "C": "-"}),
            ("C", "D", 0, {"C": ">", "D": ">"}),
        ]
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # multiedges
        edges = [("A", "B", "->"), ("A", "B", "--"), ("A", "B", "oo")]
        graph = _CoreGraph()
        graph.add_edges_from(edge_list=edges)

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 0, {"A": "-", "B": ">"}),
            ("A", "B", 1, {"A": "-", "B": "-"}),
            ("A", "B", 2, {"A": "o", "B": "o"}),
        ]
        check_graph_status(graph, 2, 3, set(), set(), set(), {})

        # fails: None node / wrong tuple length / same node / invalid edge_type
        graph = _CoreGraph()

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), (None, "A", "->"), ("B", "C", "->")]
            graph.add_edges_from(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `u`, `v` value
            edges = [("A", "B", "->"), ("A", None, "->"), ("B", "C", "->")]
            graph.add_edges_from(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError, match="3 elements"):  # 2-tuple edge (missing `edge_type`)
            edges = [("A", "B", "->"), ("A", "C"), ("B", "C", "->")]
            graph.add_edges_from(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError, match="3 elements"):  # 4-tuple edge
            edges = [("A", "B", "->"), ("A", "C", "key", "->"), ("B", "C", "->")]
            graph.add_edges_from(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # same node error
            edges = [("A", "B", "->"), ("A", "A", "->"), ("B", "C", "->")]
            graph.add_edges_from(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        with pytest.raises(ValueError):  # invalid `edge_type` value
            edges = [("A", "B", "->"), ("B", "C", "-->"), ("C", "D", "->")]
            graph.add_edges_from(edge_list=edges)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # fails: duplicate edge within a single call (consistent with `add_edge`)
        graph = _CoreGraph()
        with pytest.raises(ValueError, match="already exists"):
            graph.add_edges_from([("A", "B", "--"), ("A", "B", "--")])

    def test_remove_edge(self):
        """Test removing an edge of a `_CoreGraph`."""
        # directed edge
        edges = [("A", "B", "->"), ("B", "C", "<-")]
        graph = _CoreGraph(edge_list=edges)

        graph.remove_edge("A", "B", "->")
        graph.remove_edge("B", "C", "<-")

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

        # undirected edge
        edges = [("A", "B", "--"), ("B", "C", "--")]
        graph = _CoreGraph(edge_list=edges)

        graph.remove_edge("A", "B", "--")
        graph.remove_edge("B", "C", "--")

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

        # bidirected edge
        edges = [("A", "B", "<>"), ("B", "C", "<>")]
        graph = _CoreGraph(edge_list=edges)

        graph.remove_edge("A", "B", "<>")
        graph.remove_edge("B", "C", "<>")

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

        # unknown edge
        edges = [("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")]
        graph = _CoreGraph(edge_list=edges)

        graph.remove_edge("A", "B", "-o")
        graph.remove_edge("B", "C", "o-")
        graph.remove_edge("C", "D", "oo")

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")
        assert not graph.has_edge("C", "D")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 4, 0, set(), set(), set(), {})

        # multiedges
        edges = [("A", "B", "->"), ("A", "B", "<>"), ("A", "B", "--")]
        graph = _CoreGraph(edge_list=edges)

        graph.remove_edge("A", "B", "->")
        graph.remove_edge("A", "B", "--")

        assert not graph.has_edge("A", "B", "->")
        assert graph.has_edge("A", "B", "<>")
        assert not graph.has_edge("A", "B", "--")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 1, {"A": ">", "B": ">"}),
        ]

        check_graph_status(graph, 2, 1, set(), set(), set(), {})

        # requires explicit edge_type (no None 'remove all' shortcut)
        edges = [("A", "B", "->"), ("A", "B", "<>"), ("A", "B", "--")]
        graph = _CoreGraph(edge_list=edges)

        # edge_type is required (no default).
        with pytest.raises(TypeError):
            graph.remove_edge("A", "B")

        # None is not a valid edge_type, and the error reports the offending edge.
        with pytest.raises(ValueError, match=r"Got \('A', 'B', None\)"):
            graph.remove_edge("A", "B", None)

        # Parallel edges must be removed one explicit type at a time.
        graph.remove_edge("A", "B", "->")
        graph.remove_edge("A", "B", "<>")
        graph.remove_edge("A", "B", "--")
        assert not graph.has_edge("A", "B")
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        # fails: None node / same node / unsupported type / no such edge
        edges = [("A", "B", "->"), ("B", "C", "->")]

        graph = _CoreGraph(edge_list=edges)
        graph.remove_edge("A", "B", "->")
        with pytest.raises(ValueError):  # invalid `u`, `v` value
            graph.remove_edge(None, "C", "->")
        check_graph_status(graph, 3, 1, set(), set(), set(), {})

        graph = _CoreGraph(edge_list=edges)
        graph.remove_edge("A", "B", "->")
        with pytest.raises(ValueError):  # invalid `u`, `v` value
            graph.remove_edge("B", None, "->")
        check_graph_status(graph, 3, 1, set(), set(), set(), {})

        graph = _CoreGraph(edge_list=edges)
        graph.remove_edge("A", "B", "->")
        with pytest.raises(ValueError):  # same node error
            graph.remove_edge("B", "B", "->")
        check_graph_status(graph, 3, 1, set(), set(), set(), {})

        graph = _CoreGraph(edge_list=edges)
        graph.remove_edge("A", "B", "->")
        with pytest.raises(ValueError):  # invalid `edge_type` value
            graph.remove_edge("B", "C", "invalid_value")
        check_graph_status(graph, 3, 1, set(), set(), set(), {})

        graph = _CoreGraph(edge_list=edges)
        with pytest.raises(ValueError, match="not in graph"):  # no edge of this type between the nodes
            graph.remove_edge("A", "B", "<>")
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_remove_edges_from(self):
        """Test removing edges of a `_CoreGraph`."""
        # directed edges
        edges = [("A", "B", "->"), ("B", "C", "<-")]
        graph = _CoreGraph(edge_list=edges)

        graph.remove_edges_from(edge_list=edges)

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

        # undirected edges
        edges = [("A", "B", "--"), ("B", "C", "--")]
        graph = _CoreGraph(edge_list=edges)

        graph.remove_edges_from(edge_list=edges)

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

        # bidirected edges
        edges = [("A", "B", "<>"), ("B", "C", "<>")]
        graph = _CoreGraph(edge_list=edges)

        graph.remove_edges_from(edge_list=edges)

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 3, 0, set(), set(), set(), {})

        # unknown edges
        edges = [("A", "B", "-o"), ("B", "C", "o-"), ("C", "D", "oo")]
        graph = _CoreGraph(edge_list=edges)

        graph.remove_edges_from(edge_list=edges)

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")
        assert not graph.has_edge("C", "D")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 4, 0, set(), set(), set(), {})

        # various edge types
        edges = [("A", "B", "->"), ("B", "C", "--"), ("C", "D", "<o")]
        graph = _CoreGraph(edge_list=edges)

        graph.remove_edges_from(edge_list=edges)

        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")
        assert not graph.has_edge("C", "D")

        assert sorted(graph.edges(keys=True, data=True)) == []

        check_graph_status(graph, 4, 0, set(), set(), set(), {})

        # multiedges
        edges = [("A", "B", "->"), ("A", "B", "<>"), ("A", "B", "--")]
        graph = _CoreGraph(edge_list=edges)

        del_edges = [("A", "B", "->"), ("A", "B", "--")]
        graph.remove_edges_from(edge_list=del_edges)

        assert graph.has_edge("A", "B")
        assert not graph.has_edge("A", "B", "->")
        assert graph.has_edge("A", "B", "<>")
        assert not graph.has_edge("A", "B", "--")

        assert sorted(graph.edges(keys=True, data=True)) == [
            ("A", "B", 1, {"A": ">", "B": ">"}),
        ]

        check_graph_status(graph, 2, 1, set(), set(), set(), {})

        # fails: None node / 2-tuple / 4-tuple / same node / unsupported edge_type
        edges = [("A", "B", "->"), ("B", "C", "->")]

        graph = _CoreGraph(edge_list=edges)
        with pytest.raises(ValueError):  # invalid `u`, `v` value
            graph.remove_edges_from([("A", "B", "->"), (None, "C", "->")])
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        graph = _CoreGraph(edge_list=edges)
        with pytest.raises(ValueError):  # invalid `u`, `v` value
            graph.remove_edges_from([("A", "B", "->"), ("B", None, "->")])
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        graph = _CoreGraph(edge_list=edges)
        with pytest.raises(ValueError, match="3 elements"):  # 2-tuple edge (missing `edge_type`)
            graph.remove_edges_from([("A", "B", "->"), ("B", "C")])
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        graph = _CoreGraph(edge_list=edges)
        with pytest.raises(ValueError, match="3 elements"):  # 4-tuple edge
            graph.remove_edges_from([("A", "B", "->"), ("B", "C", "key", "->")])
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        graph = _CoreGraph(edge_list=edges)
        with pytest.raises(ValueError):  # same node error
            graph.remove_edges_from([("A", "B", "->"), ("B", "B", "->")])
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        graph = _CoreGraph(edge_list=edges)
        with pytest.raises(ValueError):  # invalid `edge_type` value
            graph.remove_edges_from([("A", "B", "->"), ("B", "C", "invalid_value")])
        check_graph_status(graph, 3, 2, set(), set(), set(), {})

    def test_copy(self):
        """Test the `copy` method of the `_CoreGraph` class."""
        # empty graph
        graph = _CoreGraph()
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # with nodes
        nodes = ["A", "B", "C"]
        graph = _CoreGraph()
        graph.add_nodes_from(nodes)
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        # with edge_list
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        graph = _CoreGraph(edge_list=edges)
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # with edges added individually
        graph = _CoreGraph()
        graph.add_edge("A", "C", "->")
        graph.add_edge("C", "B", "<-")
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        check_graph_status(graph, 3, 2, set(), set(), set(), {})

        # with attributes
        edges = [("A", "B", "->"), ("A", "B", "<>"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]

        graph = _CoreGraph(edge_list=edges, exposures=exposures, outcomes=outcomes, latents=latents)
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        check_graph_status(
            graph,
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

        # with roles
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        roles = {"test_role": ["A", "B"]}
        graph = _CoreGraph(edge_list=edges, roles=roles)
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        check_graph_status(
            graph,
            4,
            3,
            set(),
            set(),
            set(),
            {
                "test_role": ["A", "B"],
            },
        )

        # with all values
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]
        roles = {"test_role": ["B"]}
        graph = _CoreGraph(
            edge_list=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )
        graph_copy = graph.copy()

        assert graph.__eq__(graph_copy) == True
        assert graph_copy.__eq__(graph) == True

        check_graph_status(
            graph,
            4,
            3,
            {"A"},
            {"C"},
            {"D"},
            {
                "exposures": ["A"],
                "outcomes": ["C"],
                "latents": ["D"],
                "test_role": ["B"],
            },
        )

        # roles are deep-copied: adding a role to a node that already has one does not leak to the original
        graph = _CoreGraph(edge_list=[("A", "B", "->")], exposures=["A"])
        graph_copy = graph.copy()
        graph_copy.with_role("outcomes", ["A"], inplace=True)
        assert graph_copy.outcomes == {"A"} and graph.outcomes == set()

        # fails: invalid copy argument type
        graph = _CoreGraph()
        with pytest.raises(TypeError):
            graph.copy("invalid_value")

    def test_get_neighbors(self):
        """Test `get_neighbors` method of the `_CoreGraph` class."""
        # directed edges
        graph = sample_graph1(edge_type="->")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<-")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # undirected edges
        graph = sample_graph1(edge_type="--")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # bidirected edges
        graph = sample_graph1(edge_type="<>")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # unknown/partial edge marks
        graph = sample_graph1(edge_type="o>")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_neighbors("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_neighbors("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # multiedges
        graph = sample_graph3()

        all_nodes_except_A = {"B", "C", "D", "E", "F", "G", "H", "I", "J"}
        assert graph.get_neighbors("A") == all_nodes_except_A
        assert graph.get_neighbors("B") == {"A", "X"}
        assert graph.get_neighbors("C") == {"A", "Y"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # filtering by edge_type values
        graph = sample_graph1(edge_type="->")
        assert graph.get_neighbors("C", "->") == {"D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_neighbors("B", "->") == {"C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph1(edge_type="->")
        assert graph.get_neighbors("C", "<-") == {"B"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_neighbors("B", "<-") == {"A"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_neighbors("C", "o-") == {"B"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_neighbors("B", "o-") == {"A"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph3()
        assert graph.get_neighbors("A", "->") == {"B"}
        assert graph.get_neighbors("A", "<-") == {"C"}
        assert graph.get_neighbors("A", "<>") == {"E"}
        assert graph.get_neighbors("B", "->") == {"X"}
        assert graph.get_neighbors("B", "<-") == {"A"}
        assert graph.get_neighbors("C", "->") == {"A"}
        assert graph.get_neighbors("C", "<-") == {"Y"}
        assert graph.get_neighbors("C", "<>") == set()
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: no nodes / nodes but no edges / wrong node value / wrong edge_type
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_neighbors("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_neighbors("A") == set()
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_neighbors()
        with pytest.raises(ValueError):
            graph.get_neighbors(1)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")

        with pytest.raises(ValueError):
            graph.get_neighbors("A", "wrong_edge")
        check_graph_status(graph, 2, 1, set(), set(), set(), {})

    def test_get_parents(self):
        """Test `get_parents` method of the `_CoreGraph` class."""
        # directed edges
        graph = sample_graph1(edge_type="->")
        assert graph.get_parents("C") == {"B"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_parents("B") == {"A"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<-")
        assert graph.get_parents("C") == {"D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_parents("B") == {"C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # undirected edges
        graph = sample_graph1(edge_type="--")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # bidirected edges
        graph = sample_graph1(edge_type="<>")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # unknown edges
        graph = sample_graph1(edge_type="o>")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_parents("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_parents("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # multiedges
        graph = sample_graph3()
        assert graph.get_parents("A") == {"C"}
        assert graph.get_parents("B") == {"A"}
        assert graph.get_parents("C") == {"Y"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: no nodes / nodes but no edges / missing arg / unsupported type
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_parents("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_parents("A") == set()
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_parents()
        with pytest.raises(ValueError):
            graph.get_parents(1)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_get_children(self):
        """Test `get_children` method of the `_CoreGraph` class."""
        # directed edges
        graph = sample_graph1(edge_type="->")
        assert graph.get_children("C") == {"D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_children("B") == {"C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<-")
        assert graph.get_children("C") == {"B"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_children("B") == {"A"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # undirected edges
        graph = sample_graph1(edge_type="--")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # bidirected edges
        graph = sample_graph1(edge_type="<>")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # unknown edges
        graph = sample_graph1(edge_type="o>")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_children("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_children("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # multiedges
        graph = sample_graph3()
        assert graph.get_children("A") == {"B"}
        assert graph.get_children("B") == {"X"}
        assert graph.get_children("Y") == {"C"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: no nodes / nodes but no edges / unsupported input
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_children("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_children("A") == set()
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_children()
        with pytest.raises(ValueError):
            graph.get_children(1)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_get_spouses(self):
        """Test `get_spouses` method of the `_CoreGraph` class."""
        # directed edges
        graph = sample_graph1(edge_type="->")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<-")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # undirected edges
        graph = sample_graph1(edge_type="--")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # bidirected edges
        graph = sample_graph1(edge_type="<>")
        assert graph.get_spouses("C") == {"B", "D"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_spouses("B") == {"A", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # unknown edges
        graph = sample_graph1(edge_type="o>")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_spouses("C") == set()
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # multiedges
        graph = sample_graph3()
        assert graph.get_spouses("A") == {"E"}
        assert graph.get_spouses("E") == {"A"}
        assert graph.get_spouses("B") == set()
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: no nodes / nodes but no edges / unsupported input values
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_spouses("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_spouses("A") == set()
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_spouses()
        with pytest.raises(ValueError):
            graph.get_spouses(1)
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_get_ancestors(self):
        """Test `get_ancestors` method of the `_CoreGraph` class."""
        # directed edges
        graph = sample_graph1(edge_type="->")
        assert graph.get_ancestors("C") == {"A", "B", "C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_ancestors("B") == {"A", "B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<-")
        assert graph.get_ancestors("C") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_ancestors("B") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # undirected edges
        graph = sample_graph1(edge_type="--")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # bidirected edges
        graph = sample_graph1(edge_type="<>")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # unknown edges
        graph = sample_graph1(edge_type="o>")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_ancestors("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_ancestors("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # multiedges
        graph = sample_graph3()
        assert graph.get_ancestors("A") == {"A", "C", "Y"}
        assert graph.get_ancestors("X") == {"X", "B", "A", "C", "Y"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: no nodes / nodes without edges / unsupported input
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_ancestors("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_ancestors("A") == {"A"}
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_ancestors()
        with pytest.raises(ValueError):
            graph.get_ancestors("1")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_get_descendants(self):
        """Test `get_descendants` method of the `_CoreGraph` class."""
        # directed edges
        graph = sample_graph1(edge_type="->")
        assert graph.get_descendants("C") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_descendants("B") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<-")
        assert graph.get_descendants("C") == {"A", "B", "C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_descendants("B") == {"A", "B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # undirected edges
        graph = sample_graph1(edge_type="--")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # bidirected edges
        graph = sample_graph1(edge_type="<>")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # unknown edges
        graph = sample_graph1(edge_type="o>")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_descendants("C") == {"C"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_descendants("B") == {"B"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # multiedges
        graph = sample_graph3()
        assert graph.get_descendants("A") == {"A", "B", "X"}
        assert graph.get_descendants("Y") == {"Y", "C", "A", "B", "X"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: empty graph / nodes without edges / no-arg and unknown node
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_descendants("A")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_descendants("A") == {"A"}
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_descendants()
        with pytest.raises(ValueError):
            graph.get_descendants("1")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_get_reachable_nodes(self):
        """Test `get_reachable_nodes` method of the `_CoreGraph` class."""
        # directed edges
        graph = sample_graph1(edge_type="->")
        assert graph.get_reachable_nodes("C", "->") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="->")
        assert graph.get_reachable_nodes("B", "->") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<-")
        assert graph.get_reachable_nodes("C", "<-") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<-")
        assert graph.get_reachable_nodes("B", "<-") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # undirected edges
        graph = sample_graph1(edge_type="--")
        assert graph.get_reachable_nodes("C", "--") == {"A", "B", "C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="--")
        assert graph.get_reachable_nodes("B", "--") == {"A", "B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # bidirected edges
        graph = sample_graph1(edge_type="<>")
        assert graph.get_reachable_nodes("C", "<>") == {"A", "B", "C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<>")
        assert graph.get_reachable_nodes("B", "<>") == {"A", "B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # unknown / partially-directed edges
        graph = sample_graph1(edge_type="o>")
        assert graph.get_reachable_nodes("C", "o>") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="<o")
        assert graph.get_reachable_nodes("C", "<o") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="-o")
        assert graph.get_reachable_nodes("C", "-o") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="o-")
        assert graph.get_reachable_nodes("C", "o-") == {"C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph1(edge_type="oo")
        assert graph.get_reachable_nodes("C", "oo") == {"A", "B", "C", "D", "E"}
        check_graph_status(graph, 5, 4, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o>")
        assert graph.get_reachable_nodes("B", "o>") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="<o")
        assert graph.get_reachable_nodes("B", "<o") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="-o")
        assert graph.get_reachable_nodes("B", "-o") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="o-")
        assert graph.get_reachable_nodes("B", "o-") == {"B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        graph = sample_graph2(edge_type="oo")
        assert graph.get_reachable_nodes("B", "oo") == {"A", "B", "C", "D"}
        check_graph_status(graph, 4, 3, set(), set(), set(), {})

        # multiedges
        graph = sample_graph3()
        assert graph.get_reachable_nodes("A", "->") == {
            "A",
            "B",
            "X",
        }
        assert graph.get_reachable_nodes("A", "<-") == {
            "A",
            "C",
            "Y",
        }
        assert graph.get_reachable_nodes("A", "oo") == {"A", "D"}
        assert graph.get_reachable_nodes("A", "<>") == {"A", "E"}
        assert graph.get_reachable_nodes("A", "--") == {"A", "F"}
        assert graph.get_reachable_nodes("A", "o>") == {"A", "I"}
        assert graph.get_reachable_nodes("A", "<o") == {"A", "J"}
        check_graph_status(graph, 12, 11, set(), set(), set(), {})

        # fails: no nodes / nodes without edges / wrong input values
        graph = _CoreGraph()

        with pytest.raises(ValueError):
            graph.get_reachable_nodes("A", "->")
        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        assert graph.get_reachable_nodes("A", "->") == {"A"}
        check_graph_status(graph, 2, 0, set(), set(), set(), {})

        graph = _CoreGraph()

        with pytest.raises(TypeError):
            graph.get_reachable_nodes()
        with pytest.raises(ValueError):
            graph.get_reachable_nodes("A")

        check_graph_status(graph, 0, 0, set(), set(), set(), {})

    def test_get_edges(self):
        # all edges with and without data
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "o-")
        graph.add_edge("A", "B", "<>")

        assert sorted(graph.get_edges(data=False)) == sorted(
            [
                ("A", "B"),
                ("A", "B"),
                ("B", "C"),
            ]
        )
        assert sorted(graph.get_edges(data=True)) == sorted(
            [
                ("A", "B", "->"),
                ("A", "B", "<>"),
                ("B", "C", "o-"),
            ]
        )

    def test_get_edge(self):
        # edge retrieval: directed, bidirected, partially-directed, circle, and reversed lookups
        graph = _CoreGraph()

        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "--")
        graph.add_edge("B", "C", "<>")
        graph.add_edge("C", "D", "-o")
        graph.add_edge("D", "E", "<-")
        graph.add_edge("E", "F", "oo")
        graph.add_edge("F", "G", "o-")
        graph.add_node("H")

        assert set(graph.get_edge("A", "B", data=True)) == {
            ("A", "B", "->"),
            ("A", "B", "--"),
        }
        assert set(graph.get_edge("B", "C", data=True)) == {
            ("B", "C", "<>"),
        }
        assert set(graph.get_edge("D", "E")) == {
            ("D", "E", "<-"),
        }
        assert set(graph.get_edge("E", "D")) == {
            ("E", "D", "->"),
        }
        assert set(graph.get_edge("E", "F")) == {
            ("E", "F", "oo"),
        }
        assert set(graph.get_edge("F", "G")) == {
            ("F", "G", "o-"),
        }

        # fails: no edge exists between the two nodes
        graph = _CoreGraph()
        graph.add_node("A")
        graph.add_node("B")

        with pytest.raises(ValueError):
            graph.get_edge("A", "B")

    def test_has_edge(self):
        # has_edge: directed, undirected, bidirected, reversed forms, and missing edges
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "--")
        graph.add_edge("C", "D", "<>")
        graph.add_node("E")

        assert graph.has_edge("A", "B") is True
        assert graph.has_edge("A", "B", "->") is True
        assert graph.has_edge("B", "C", "--") is True
        assert graph.has_edge("C", "D", "<>") is True
        assert graph.has_edge("D", "C", "<>") is True
        assert graph.has_edge("B", "A", "<-") is True
        assert graph.has_edge("A", "B", "<-") is False
        assert graph.has_edge("A", "B", "--") is False
        assert graph.has_edge("A", "B", "<>") is False
        assert graph.has_edge("A", "B", "-o") is False
        assert graph.has_edge("A", "E") is False
        assert graph.has_edge("A", "E", "->") is False

        # fails: unsupported edge type
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("B", "C", "--")
        graph.add_edge("C", "D", "<>")
        graph.add_node("E")

        with pytest.raises(ValueError):
            graph.has_edge("A", "B", "invalid_edge_type")

    def test_replace_edge(self):
        # replace directed/bidirected/circle edge types in sequence
        graph = _CoreGraph()
        graph.add_edge("A", "B", "--")

        graph.replace_edge("A", "B", old_type="--", new_type="->")
        assert graph.get_edges(data=True) == (
            [
                ("A", "B", "->"),
            ]
        )

        graph.replace_edge("A", "B", old_type="->", new_type="<>")
        assert graph.get_edges(data=True) == (
            [
                ("A", "B", "<>"),
            ]
        )

        graph.replace_edge("A", "B", old_type="<>", new_type="oo")
        assert graph.get_edges(data=True) == (
            [
                ("A", "B", "oo"),
            ]
        )

        # fails: invalid old_type / invalid new_type / nonexistent edge
        graph = _CoreGraph()
        graph.add_node("C")
        graph.add_edge("A", "B", "--")

        with pytest.raises(ValueError):
            graph.replace_edge("A", "B", old_type="!!", new_type="->")

        with pytest.raises(ValueError):
            graph.replace_edge("A", "B", old_type="--", new_type="~~")

        with pytest.raises(ValueError):
            graph.replace_edge("B", "C", old_type="--", new_type="->")

    def test_eq(self):
        """Test the `__eq__` method of the `_CoreGraph` class."""
        # empty graphs are equal
        graph = _CoreGraph()
        other = _CoreGraph()

        assert graph.__eq__(other) == True
        assert other.__eq__(graph) == True

        check_graph_status(graph, 0, 0, set(), set(), set(), {})

        # graphs with identical values are equal
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]
        roles = {"test_role": ["B"]}
        graph = _CoreGraph(
            edge_list=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

        other = _CoreGraph()
        other.add_edges_from(edge_list=edges)
        other.exposures = exposures
        other.outcomes = outcomes
        other.latents = latents
        other.with_role("test_role", ["B"], inplace=True)

        assert graph.__eq__(other) == True
        assert other.__eq__(graph) == True

        check_graph_status(
            graph,
            4,
            3,
            {"A"},
            {"C"},
            {"D"},
            {
                "exposures": ["A"],
                "outcomes": ["C"],
                "latents": ["D"],
                "test_role": ["B"],
            },
        )

        # fails: no argument / too many arguments / called on non-graph
        graph = _CoreGraph()
        with pytest.raises(TypeError):
            graph.__eq__()
        with pytest.raises(TypeError):
            graph.__eq__(_CoreGraph(), _CoreGraph())

        with pytest.raises(AssertionError):
            # Not a graph class
            other_str = "not a graph"
            assert other_str.__eq__(graph) == False

        # different graphs are not equal (class / edges / roles / non-graph)
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "oo")]
        exposures = ["A"]
        outcomes = ["C"]
        latents = ["D"]
        roles = {"test_role": ["B"]}
        graph = _CoreGraph(
            edge_list=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )

        # Different class
        other_dag = DAG()
        other_admg = ADMG()
        other_pdag = PDAG()

        assert graph.__eq__(other_dag) == False
        assert other_dag.__eq__(graph) == False
        assert graph.__eq__(other_admg) == False
        assert other_admg.__eq__(graph) == False
        assert graph.__eq__(other_pdag) == False
        assert other_pdag.__eq__(graph) == False

        # Different edge_list
        other_edge_list = _CoreGraph(
            edge_list=[
                ("A", "B", "->"),
                ("B", "C", "<>"),
                ("B", "C", "->"),
                ("C", "D", "oo"),
            ],
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )
        assert graph.__eq__(other_edge_list) == False
        assert other_edge_list.__eq__(graph) == False

        # Different exposures
        other_exp = _CoreGraph(
            edge_list=edges,
            exposures=["B"],
            outcomes=outcomes,
            latents=latents,
            roles=roles,
        )
        assert graph.__eq__(other_exp) == False
        assert other_exp.__eq__(graph) == False

        # Different outcomes
        other_out = _CoreGraph(
            edge_list=edges,
            exposures=exposures,
            outcomes=["A"],
            latents=latents,
            roles=roles,
        )
        assert graph.__eq__(other_out) == False
        assert other_out.__eq__(graph) == False

        # Different latents
        other_lat = _CoreGraph(
            edge_list=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=["B"],
            roles=roles,
        )
        assert graph.__eq__(other_lat) == False
        assert other_lat.__eq__(graph) == False

        # Different roles
        other_roles = _CoreGraph(
            edge_list=edges,
            exposures=exposures,
            outcomes=outcomes,
            latents=latents,
            roles={"new_role": ["A"]},
        )
        assert graph.__eq__(other_roles) == False
        assert other_roles.__eq__(graph) == False

        # Not a graph class
        other_str = "not a graph"
        assert graph.__eq__(other_str) == False

    def test_to_markers(self):
        # cases: circle-line / arrow-circle / bidirected / undirected / directed edges
        edge_tuple = ("A", "B", "o-")
        graph = _CoreGraph()
        assert graph._to_markers(edge_tuple) == {"B": "-", "A": "o"}

        edge_tuple = ("A", "B", "<o")
        graph = _CoreGraph()
        assert graph._to_markers(edge_tuple) == {"B": "o", "A": ">"}

        edge_tuple = ("A", "B", "<>")
        graph = _CoreGraph()
        assert graph._to_markers(edge_tuple) == {"A": ">", "B": ">"}

        edge_tuple = ("A", "B", "--")
        graph = _CoreGraph()
        assert graph._to_markers(edge_tuple) == {"A": "-", "B": "-"}

        edge_tuple = ("A", "B", "->")
        graph = _CoreGraph()
        assert graph._to_markers(edge_tuple) == {"A": "-", "B": ">"}

        # fails: invalid edge tuple with extra element
        invalid_edge = ("A", "B", "key", "<-")
        graph = _CoreGraph()
        with pytest.raises(ValueError):
            graph._to_markers(invalid_edge)

    def test_to_edge_type(self):
        # cases: reverse/circle-line/arrow-circle/bidirected/directed/undirected edge types
        # 1. Reverse Arrow Edge (Explicit check: u='>', v='-')
        u, v = "B", "A"
        markers = {"B": ">", "A": "-"}
        graph = _CoreGraph()
        assert graph._to_edge_type(u, v, markers) == "<-"

        # 2. Circle-Line Edge (Explicit check: u='o', v='-')
        u, v = "A", "B"
        markers = {"A": "o", "B": "-"}
        graph = _CoreGraph()
        assert graph._to_edge_type(u, v, markers) == "o-"

        # 3. Arrow-Circle Edge (Explicit check: u='>', v='o')
        u, v = "A", "B"
        markers = {"A": ">", "B": "o"}
        graph = _CoreGraph()
        assert graph._to_edge_type(u, v, markers) == "<o"

        # 4. Bidirected Edge (Explicit check: u='>', v='>')
        u, v = "A", "B"
        markers = {"A": ">", "B": ">"}
        graph = _CoreGraph()
        assert graph._to_edge_type(u, v, markers) == "<>"

        # 5. Directed Edge (Forward - General case via else block)
        u, v = "A", "B"
        markers = {"A": "-", "B": ">"}
        graph = _CoreGraph()
        assert graph._to_edge_type(u, v, markers) == "->"

        # 6. Undirected Edge (General case via else block)
        u, v = "A", "B"
        markers = {"A": "-", "B": "-"}
        graph = _CoreGraph()

        assert graph._to_edge_type(u, v, markers) == "--"

        # fails: missing marker for node 'u' or 'v'
        u, v = "A", "B"
        markers = {"A": "-"}  # 'B' is missing
        graph = _CoreGraph()

        with pytest.raises(KeyError):
            graph._to_edge_type(u, v, markers)

    def test_to_adjacency(self):
        """Test the `_CoreGraph.to_adjacency` method across encodings."""
        # edge_type (default): each cell is the edge type oriented row -> column
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>"), ("C", "D", "--"), ("D", "E", "o>")])
        adj = graph.to_adjacency()  # default encoding="edge_type"
        assert adj.loc["A", "B"] == "->"
        assert adj.loc["B", "A"] == "<-"
        assert adj.loc["B", "C"] == "<>"
        assert adj.loc["C", "B"] == "<>"
        assert adj.loc["C", "D"] == "--"
        assert adj.loc["D", "E"] == "o>"
        assert adj.loc["E", "D"] == "<o"
        assert adj.loc["A", "C"] == 0  # no edge
        assert adj.loc["A", "A"] == 0  # diagonal

        # edge_type: coincident directed + bidirected (ADMG) -> aligned tuple of edge types
        # (zipping the two cells reconstructs each edge from both orientations)
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        adj = graph.to_adjacency()
        assert set(zip(adj.loc["A", "B"], adj.loc["B", "A"])) == {("->", "<-"), ("<>", "<>")}

        # binary: only supported for DAG/PDAG-style classes (directed/undirected edges); a class that
        # also supports bidirected/circle edges (here base _CoreGraph) is rejected on SUPPORTED_EDGE_TYPES.
        with pytest.raises(ValueError, match="DAG and PDAG"):
            _CoreGraph(edge_list=[("A", "B", "->")]).to_adjacency(encoding="binary")

        class _DirectedGraph(_CoreGraph):
            SUPPORTED_EDGE_TYPES = frozenset({"->", "<-", "--"})

        # binary: directed -> asymmetric 0/1; undirected -> symmetric
        graph = _DirectedGraph(edge_list=[("A", "B", "->"), ("A", "C", "->"), ("B", "C", "->"), ("D", "E", "--")])
        adj = graph.to_adjacency(encoding="binary")
        assert adj.loc["A", "B"] == 1 and adj.loc["B", "A"] == 0  # arc A->B (asymmetric)
        assert adj.loc["A", "C"] == 1 and adj.loc["B", "C"] == 1
        assert adj.loc["D", "E"] == 1 and adj.loc["E", "D"] == 1  # undirected (symmetric)

        # bnlearn is accepted as an alias of binary
        assert graph.to_adjacency(encoding="bnlearn").equals(graph.to_adjacency(encoding="binary"))

        # binary: coincident edges (e.g. a directed and an undirected edge between a pair) -> raise
        coincident = _DirectedGraph()
        coincident.add_edge("A", "B", "->")
        coincident.add_edge("A", "B", "--")
        with pytest.raises(ValueError, match="coincident"):
            coincident.to_adjacency(encoding="binary")

        # causal-learn: integer codes, mark at the ROW endpoint
        graph = _CoreGraph(edge_list=[("A", "B", "->")])
        adj = graph.to_adjacency(encoding="causal-learn")
        assert adj.loc["A", "B"] == -1 and adj.loc["B", "A"] == 1

        # causal-learn: coincident -> composite codes 4 / 5
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        adj = graph.to_adjacency(encoding="causal-learn")
        assert adj.loc["A", "B"] == 4 and adj.loc["B", "A"] == 5

        # pcalg amat.pag: column-indexed (1=circle, 2=arrowhead, 3=tail)
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("C", "D", "o>")])
        adj = graph.to_adjacency(encoding="pcalg")
        assert adj.loc["A", "B"] == 2 and adj.loc["B", "A"] == 3  # arrow at B, tail at A
        assert adj.loc["C", "D"] == 2 and adj.loc["D", "C"] == 1  # arrow at D, circle at C

        # pcalg: coincident edges cannot be represented -> raise
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        with pytest.raises(ValueError, match="pcalg"):
            graph.to_adjacency(encoding="pcalg")

        # fails: unknown encoding
        with pytest.raises(ValueError, match="encoding"):
            _CoreGraph(edge_list=[("A", "B", "->")]).to_adjacency(encoding="nonsense")

    def test_get_subgraph(self):
        """Test the `_CoreGraph.get_subgraph` method (class-preserving, node-induced, independent copy)."""
        # node-induced subgraph keeps only edges among the requested nodes, with their edge types preserved
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "<>"), ("C", "D", "--"), ("D", "E", "o>")])
        sub = graph.get_subgraph(["A", "B", "C"])
        assert type(sub) is _CoreGraph
        assert set(sub.nodes()) == {"A", "B", "C"}
        assert set(sub.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "<>")}

        # the result is an independent copy, not a networkx view: mutating one does not affect the other
        sub.add_edge("A", "C", "->")
        assert not graph.has_edge("A", "C")
        graph.remove_edge("A", "B", "->")
        assert sub.has_edge("A", "B", "->")

        # roles (exposures/outcomes/latents/custom) are carried over, restricted to the retained nodes
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "->")]
        graph = _CoreGraph(edge_list=edges, exposures=["A"], outcomes=["C"], latents=["D"])
        sub = graph.get_subgraph(["A", "B", "C"])
        check_graph_status(sub, 3, 2, {"A"}, {"C"}, set(), {"exposures": ["A"], "outcomes": ["C"]})

        # roles are deep-copied: adding a role to a node that already has one does not leak to the original
        sub.with_role("outcomes", ["A"], inplace=True)
        assert sub.outcomes == {"A", "C"} and graph.outcomes == {"C"}

        # a single-node list yields a one-node subgraph with no edges
        sub = graph.get_subgraph(["A"])
        assert set(sub.nodes()) == {"A"}
        assert sub.get_edges(data=True) == []

        # coincident edges (a directed and a bidirected edge between the same pair) are both preserved
        graph = _CoreGraph()
        graph.add_edge("A", "B", "->")
        graph.add_edge("A", "B", "<>")
        graph.add_edge("B", "C", "->")
        sub = graph.get_subgraph(["A", "B"])
        assert set(sub.get_edge("A", "B")) == {("A", "B", "->"), ("A", "B", "<>")}
        assert not sub.has_edge("B", "C")

        # the subclass is preserved (here ADMG)
        admg = ADMG(edge_list=[("A", "B", "->"), ("B", "C", "<>"), ("C", "D", "->")])
        sub = admg.get_subgraph(["B", "C", "D"])
        assert type(sub) is ADMG
        assert set(sub.get_edges(data=True)) == {("B", "C", "<>"), ("C", "D", "->")}

        # requesting a node that is not in the graph raises
        with pytest.raises(ValueError, match="not in graph"):
            admg.get_subgraph(["B", "Z"])

        # get_ancestral_graph (which is built on get_subgraph) still returns the correct independent graph
        graph = _CoreGraph(edge_list=[("A", "B", "->"), ("B", "C", "->"), ("C", "D", "<>"), ("C", "E", "--")])
        ancestral = graph.get_ancestral_graph("C")
        assert type(ancestral) is _CoreGraph
        assert set(ancestral.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}
        ancestral.add_edge("A", "C", "->")
        assert not graph.has_edge("A", "C")
