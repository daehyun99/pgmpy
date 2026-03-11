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
    return MAG(ebunch=edges, roles=roles)


@pytest.fixture
def mag2():
    edges = [
        ("P", "Q", "<>"),
        ("Q", "R", "->"),
        ("P", "R", "->"),
        ("P", "L", "->"),
    ]
    return MAG(ebunch=edges, latents={"L"})


# mag3 and mag4 are taken from Maathuis 2018 JMLR Figure 2
@pytest.fixture
def mag3():
    edges = [("V", "X", "->"), ("X", "Y", "->")]
    return MAG(ebunch=edges)


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
    return MAG(ebunch=edges)


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
        m1 = MAG(ebunch=e, latents={"L"}, roles=roles)
        m2 = MAG(
            ebunch=e,
            latents={"L"},
            roles={"exposures": "X", "outcomes": "Z", "adjustment": {"Y"}},
        )
        assert m1 == m2

        m3 = MAG(ebunch=e, latents={"L"}, roles={"exposures": "X"})
        assert m1 != m3

        m4 = MAG(
            ebunch=[
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

        m5 = MAG(ebunch=e, latents={"L", "U"}, roles=roles)
        assert m1 != m5

    @pytest.mark.skip(
        reason="Refactoring: Skip for evaluation integration into _GraphAlgorithmMixin class. (Related: #2384, #2385)"
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
        reason="Refactoring: Skip for evaluation integration into _GraphAlgorithmMixin class. (Related: #2384, #2385)"
    )
    def test_from_dagitty_disconnected_graphs(self):
        model_str = """
            mag {
                "Wet grass" [exposure]
                'Large Name' <-> Node ; Rain -> "Wet grass"
                Node [o]
            }"""

        model_from_str = MAG.from_dagitty(model_str)

        expected_nodes = {"Large Name", "Node", "Rain", "Wet grass"}
        expected_roles = {"outcomes": ["Node"], "exposures": ["Wet grass"]}

        assert set(model_from_str.nodes()) == expected_nodes
        assert model_from_str.get_role_dict() == expected_roles
