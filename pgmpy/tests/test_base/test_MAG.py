import os

import pytest

from pgmpy.base import MAG


# graph has been taken from the zhang 2008 paper (figure 1)
@pytest.fixture
def mag():
    edges = [
        ("A", "B", "<>"),
        ("C", "D", "<>"),
        ("A", "C", "<>"),
        ("B", "D", "<>"),
        ("A", "D", "->"),
        ("B", "C", "->"),
    ]
    roles = {"exposures": {"A"}, "outcomes": {"D"}, "adjustment": {"B", "C"}}
    return MAG(edge_list=edges, roles=roles)


@pytest.fixture
def mag2():
    edges = [
        ("P", "Q", "<>"),
        ("Q", "R", "->"),
        ("P", "R", "->"),
        ("P", "L", "->"),
    ]
    return MAG(edge_list=edges, latents={"L"})


# mag3 and mag4 are taken from Maathuis 2018 JMLR Figure 2
@pytest.fixture
def mag3():
    edges = [("V", "X", "->"), ("X", "Y", "->")]
    return MAG(edge_list=edges)


@pytest.fixture
def mag4():
    edges = [
        ("V1", "V2", "<>"),
        ("V2", "V3", "<>"),
        ("V3", "V4", "<>"),
        ("V4", "X", "<>"),
        ("X", "Y", "->"),
        ("V2", "Y", "->"),
        ("V3", "Y", "->"),
        ("V4", "Y", "->"),
    ]
    return MAG(edge_list=edges)


class TestMAG:
    def test_empty_init(self):
        empty = MAG()
        assert len(empty.nodes()) == 0
        assert empty.latents == set()

    def test_roles_and_equality(self):
        e = [
            ("X", "Z", "->"),
            ("Y", "Z", "->"),
            ("L", "X", "->"),
            ("L", "Z", "->"),
            ("U", "X", "->"),
        ]
        roles = {"exposures": "X", "outcomes": "Z", "adjustment": {"Y"}}
        m1 = MAG(edge_list=e, latents={"L"}, roles=roles)
        m2 = MAG(
            edge_list=e,
            latents={"L"},
            roles={"exposures": "X", "outcomes": "Z", "adjustment": {"Y"}},
        )
        assert m1 == m2

        m3 = MAG(edge_list=e, latents={"L"}, roles={"exposures": "X"})
        assert m1 != m3

        m4 = MAG(
            edge_list=[
                ("X", "Z", "<>"),
                ("Y", "Z", "->"),
                ("L", "X", "->"),
                ("L", "Z", "->"),
                ("U", "X", "->"),
            ],
            latents={"L"},
            roles=roles,
        )
        assert m1 != m4

        m5 = MAG(edge_list=e, latents={"L", "U"}, roles=roles)
        assert m1 != m5

    @pytest.mark.skip(
        reason="Refactoring: Skip for evaluation integration into _GraphAlgorithms class. (Related: #2384, #2385)"
    )
    def test_from_dagitty(self):
        model_str = "mag { E [latent] A [e] J [o] {B, E} -> A; A -- J ; A -- M}"
        model_from_str = MAG.from_dagitty(model_str)
        with open("test_model.dagitty", "w") as f:
            f.write(model_str)
        model_from_file = MAG.from_dagitty(filename="test_model.dagitty")
        os.remove("test_model.dagitty")

        expected_edges = {("B", "A"), ("A", "E"), ("A", "J"), ("A", "M")}
        expected_roles = {"outcomes": ["J"], "latents": ["E"], "exposures": ["A"]}

        assert model_from_str.edges() == expected_edges
        assert model_from_str.get_role_dict() == expected_roles
        assert model_from_file.edges() == expected_edges
        assert model_from_file.get_role_dict() == expected_roles

    @pytest.mark.skip(
        reason="Refactoring: Skip for evaluation integration into _GraphAlgorithms class. (Related: #2384, #2385)"
    )
    def test_from_dagitty_disconnected_graphs(self):
        model_str = """
            mag {
                "Wet grass" [exposure]
                'Large Name' <-> Node ; Rain -> "Wet grass"
                Node [outcome]
            }"""

        model_from_str = MAG.from_dagitty(model_str)

        expected_nodes = {"Large Name", "Node", "Rain", "Wet grass"}
        expected_roles = {"outcomes": ["Node"], "exposures": ["Wet grass"]}

        assert set(model_from_str.nodes()) == expected_nodes
        assert model_from_str.get_role_dict() == expected_roles

    @pytest.mark.skip(reason="Refactoring: Skip now. I implement this When Refactoring DAG(Related: #2384, #2385)")
    def test_is_valid_mag(self):
        """Test code for `is_valid_mag` method"""
        # TODO(@daehyun99): [#2384] Implement code logic and test code
        ...

    def test_add_directed_edge(self):
        """Test adding directed edges."""
        mag = MAG()
        mag.add_edge("A", "B", "->")
        mag.add_edge("B", "C", "->")

        assert mag.has_edge("A", "B")
        assert set(mag.get_edges(data=True)) == {("A", "B", "->"), ("B", "C", "->")}

    def test_add_undirected_edge(self):
        """Test adding undirected edges."""
        mag = MAG()
        mag.add_edge("A", "B", "--")
        mag.add_edge("B", "C", "--")

        assert mag.has_edge("A", "B")
        assert set(mag.get_edges(data=True)) == {("A", "B", "--"), ("B", "C", "--")}

    def test_add_bidirected_edge(self):
        """Test adding bidirected edges."""
        mag = MAG()
        mag.add_edge("X", "Y", "<>")

        assert mag.has_edge("X", "Y")
        assert set(mag.get_edges(data=True)) == {("X", "Y", "<>")}

    def test_add_directed_edges(self):
        """Test adding multiple directed edges at once."""
        mag = MAG()
        edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "->")]
        mag.add_edges_from(edges)

        for u, v, _ in edges:
            assert mag.has_edge(u, v)

        assert set(mag.get_edges(data=True)) == set(edges)

    def test_add_undirected_edges(self):
        """Test adding multiple undirected edges at once."""
        mag = MAG()
        edges = [("A", "B", "--"), ("B", "C", "--"), ("C", "D", "--")]
        mag.add_edges_from(edges)

        for u, v, _ in edges:
            assert mag.has_edge(u, v)

        assert set(mag.get_edges(data=True)) == set(edges)

    def test_add_bidirected_edges(self):
        """Test adding multiple bidirected edges at once."""
        mag = MAG()
        edges = [("X", "Y", "<>"), ("Y", "Z", "<>")]
        mag.add_edges_from(edges)

        for u, v, _ in edges:
            assert mag.has_edge(u, v)

        assert set(mag.get_edges(data=True)) == set(edges)
