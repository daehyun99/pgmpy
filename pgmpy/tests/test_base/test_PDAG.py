import unittest

from pgmpy.base import DAG, PDAG


class TestPDAG(unittest.TestCase):
    def test_init_normal(self):
        # Mix directed and undirected
        ebunch_mix = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag = PDAG(ebunch=ebunch_mix)
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(
            sorted(pdag.get_edges(keys=False, data=True)),
            [("A", "B", "--"), ("A", "C", "->"), ("C", "D", "<-"), ("D", "B", "--")],
        )

        pdag = PDAG(
            ebunch=ebunch_mix,
            latents=["A", "C"],
        )
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(
            sorted(pdag.get_edges(keys=False, data=True)),
            [("A", "B", "--"), ("A", "C", "->"), ("C", "D", "<-"), ("D", "B", "--")],
        )
        self.assertEqual(pdag.latents, {"A", "C"})

        # Only undirected
        ebunch_undir = [("A", "C", "--"), ("D", "C", "--"), ("B", "A", "--"), ("B", "D", "--")]
        pdag = PDAG(ebunch=ebunch_undir)
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(
            sorted(pdag.get_edges(keys=False, data=True)),
            [("A", "B", "--"), ("A", "C", "--"), ("C", "D", "--"), ("D", "B", "--")],
        )

        pdag = PDAG(ebunch=ebunch_undir, latents=["A", "D"])
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(
            sorted(pdag.get_edges(keys=False, data=True)),
            [("A", "B", "--"), ("A", "C", "--"), ("C", "D", "--"), ("D", "B", "--")],
        )
        self.assertEqual(pdag.latents, {"A", "D"})

        # Only directed
        ebunch_dir = [("A", "B", "->"), ("D", "B", "->"), ("A", "C", "->"), ("D", "C", "->")]
        pdag = PDAG(ebunch=ebunch_dir)
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(
            sorted(pdag.get_edges(keys=False, data=True)),
            [("A", "B", "->"), ("A", "C", "->"), ("B", "D", "<-"), ("D", "C", "->")],
        )

        pdag = PDAG(ebunch=ebunch_dir, latents=["D"])
        self.assertEqual(set(pdag.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(
            sorted(pdag.get_edges(keys=False, data=True)),
            [("A", "B", "->"), ("A", "C", "->"), ("B", "D", "<-"), ("D", "C", "->")],
        )
        self.assertEqual(pdag.latents, {"D"})

    def test_all_neighrors(self):
        pdag = PDAG(ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")])

        self.assertEqual(pdag.get_neighbors(node="A"), {"B", "C"})
        self.assertEqual(pdag.get_neighbors(node="B"), {"A", "D"})
        self.assertEqual(pdag.get_neighbors(node="C"), {"A", "D"})
        self.assertEqual(pdag.get_neighbors(node="D"), {"B", "C"})

    def test_get_children(self):
        pdag = PDAG(ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")])

        self.assertEqual(pdag.get_children(node="A"), {"C"})
        self.assertEqual(pdag.get_children(node="B"), set())
        self.assertEqual(pdag.get_children(node="C"), set())

    def test_get_parents(self):
        pdag = PDAG(ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")])

        self.assertEqual(pdag.get_parents(node="A"), set())
        self.assertEqual(pdag.get_parents(node="B"), set())
        self.assertEqual(pdag.get_parents(node="C"), {"A", "D"})

    def test_get_neighbors(self):
        pdag = PDAG(ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")])

        self.assertEqual(pdag.get_neighbors(node="A", edge_type="--"), {"B"})
        self.assertEqual(pdag.get_neighbors(node="B", edge_type="--"), {"A", "D"})
        self.assertEqual(pdag.get_neighbors(node="C", edge_type="--"), set())
        self.assertEqual(pdag.get_neighbors(node="D", edge_type="--"), {"B"})

    def test_orient_undirected_edge(self):
        pdag = PDAG(ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")])

        mod_pdag = pdag.orient_undirected_edge("B", "A", inplace=False)
        self.assertEqual(
            sorted(mod_pdag.get_edges(keys=False, data=True)),
            [("A", "B", "<-"), ("A", "C", "->"), ("C", "D", "<-"), ("D", "B", "--")],
        )

        pdag.orient_undirected_edge("B", "A", inplace=True)
        self.assertEqual(
            sorted(pdag.get_edges(keys=False, data=True)),
            [("A", "B", "<-"), ("A", "C", "->"), ("C", "D", "<-"), ("D", "B", "--")],
        )

        self.assertRaises(ValueError, pdag.orient_undirected_edge, "B", "A", inplace=True)

    def test_copy(self):
        ebunch = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag_mix = PDAG(ebunch)
        pdag_copy = pdag_mix.copy()
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(
            sorted(pdag_copy.get_edges(keys=False, data=True)),
            [("A", "B", "--"), ("A", "C", "->"), ("C", "D", "<-"), ("D", "B", "--")],
        )
        self.assertEqual(pdag_copy.latents, set())

        ebunch = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag_latent = PDAG(
            ebunch,
            latents=["A", "D"],
        )
        pdag_copy = pdag_latent.copy()
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(
            sorted(pdag_copy.get_edges(keys=False, data=True)),
            [("A", "B", "--"), ("A", "C", "->"), ("C", "D", "<-"), ("D", "B", "--")],
        )
        self.assertEqual(pdag_copy.latents, {"A", "D"})

        ebunch = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag_role = PDAG(
            ebunch,
            roles={"exposures": "A", "adjustment": "D", "outcomes": "C"},
        )
        pdag_copy = pdag_role.copy()
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(
            sorted(pdag_copy.get_edges(keys=False, data=True)),
            [("A", "B", "--"), ("A", "C", "->"), ("C", "D", "<-"), ("D", "B", "--")],
        )
        self.assertEqual(pdag_copy.latents, set())
        self.assertEqual(pdag_copy.get_role("exposures"), ["A"])
        self.assertEqual(pdag_copy.get_role("adjustment"), ["D"])
        self.assertEqual(pdag_copy.get_role("outcomes"), ["C"])
        self.assertEqual(
            sorted(pdag_copy.get_roles()),
            sorted(["adjustment", "exposures", "outcomes"]),
        )

        ebunch = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag_role_set = PDAG(
            ebunch,
            roles={"exposures": ("A", "D"), "outcomes": ("C")},
        )
        pdag_copy = pdag_role_set.copy()
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(
            sorted(pdag_copy.get_edges(keys=False, data=True)),
            [("A", "B", "--"), ("A", "C", "->"), ("C", "D", "<-"), ("D", "B", "--")],
        )
        self.assertEqual(pdag_copy.latents, set())
        self.assertEqual(sorted(pdag_copy.get_role("exposures")), sorted(["A", "D"]))
        self.assertEqual(pdag_copy.get_role("outcomes"), ["C"])
        self.assertEqual(sorted(pdag_copy.get_roles()), sorted(["exposures", "outcomes"]))

        ebunch = [("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")]
        pdag_role_list = PDAG(
            ebunch,
            roles={"exposures": ["A", "D"], "outcomes": ["C"]},
        )
        pdag_copy = pdag_role_list.copy()
        self.assertEqual(set(pdag_copy.nodes()), {"A", "B", "C", "D"})
        self.assertEqual(
            sorted(pdag_copy.get_edges(keys=False, data=True)),
            [("A", "B", "--"), ("A", "C", "->"), ("C", "D", "<-"), ("D", "B", "--")],
        )
        self.assertEqual(pdag_copy.latents, set())
        self.assertEqual(sorted(pdag_copy.get_role("exposures")), sorted(["A", "D"]))
        self.assertEqual(pdag_copy.get_role("outcomes"), ["C"])
        self.assertEqual(sorted(pdag_copy.get_roles()), sorted(["exposures", "outcomes"]))

    def test_pdag_to_dag(self):
        # PDAG no: 1  Possibility of creating a v-structure
        pdag = PDAG(ebunch=[("A", "B", "->"), ("C", "B", "->"), ("C", "D", "--"), ("D", "A", "--")])
        dag = pdag.to_dag()
        self.assertTrue(("A", "B") in dag.edges())
        self.assertTrue(("C", "B") in dag.edges())
        self.assertFalse((("A", "D") in dag.edges()) and (("C", "D") in dag.edges()))
        self.assertTrue(len(dag.edges()) == 4)

        # With latents
        pdag = PDAG(
            ebunch=[("A", "B", "->"), ("C", "B", "->"), ("C", "D", "--"), ("D", "A", "--")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        self.assertTrue(("A", "B") in dag.edges())
        self.assertTrue(("C", "B") in dag.edges())
        self.assertFalse((("A", "D") in dag.edges()) and (("C", "D") in dag.edges()))
        self.assertEqual(dag.latents, {"A"})
        self.assertTrue(len(dag.edges()) == 4)

        # PDAG no: 2  No possibility of creation of v-structure.
        pdag = PDAG(ebunch=[("B", "C", "->"), ("A", "C", "->"), ("A", "D", "--")])
        dag = pdag.to_dag()
        self.assertTrue(("B", "C") in dag.edges())
        self.assertTrue(("A", "C") in dag.edges())
        self.assertTrue((("A", "D") in dag.edges()) or (("D", "A") in dag.edges()))

        # With latents
        pdag = PDAG(
            ebunch=[("B", "C", "->"), ("A", "C", "->"), ("A", "D", "--")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        self.assertTrue(("B", "C") in dag.edges())
        self.assertTrue(("A", "C") in dag.edges())
        self.assertTrue((("A", "D") in dag.edges()) or (("D", "A") in dag.edges()))
        self.assertEqual(dag.latents, {"A"})

        # PDAG no: 3  Already existing v-structure, possibility to add another
        pdag = PDAG(ebunch=[("B", "C", "->"), ("A", "C", "->"), ("C", "D", "--")])
        dag = pdag.to_dag()
        expected_edges = {("B", "C"), ("C", "D"), ("A", "C")}
        self.assertEqual(expected_edges, set(dag.edges()))

        # With latents
        pdag = PDAG(
            ebunch=[("B", "C", "->"), ("A", "C", "->"), ("C", "D", "--")],
            latents=["A"],
        )
        dag = pdag.to_dag()
        expected_edges = {("B", "C"), ("C", "D"), ("A", "C")}
        self.assertEqual(expected_edges, set(dag.edges()))
        self.assertEqual(dag.latents, {"A"})

        ebunch = [
            (1, 4, "--"),
            (5, 0, "--"),
            (0, 2, "->"),
            (1, 2, "->"),
            (3, 1, "->"),
            (3, 2, "->"),
            (3, 4, "->"),
            (4, 2, "->"),
            (5, 1, "->"),
            (5, 2, "->"),
            (5, 4, "->"),
        ]
        pdag = PDAG(ebunch=ebunch)
        dag = pdag.to_dag()
        dag_actual = {
            (0, 2),
            (1, 2),
            (3, 1),
            (3, 2),
            (3, 4),
            (4, 1),
            (4, 2),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 4),
        }
        self.assertSetEqual(set(dag.edges), dag_actual)

    def test_pdag_to_cpdag(self):
        pdag = PDAG(ebunch=[("A", "B", "->"), ("B", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C")})

        pdag = PDAG(ebunch=[("A", "B", "->"), ("B", "C", "--"), ("C", "D", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C"), ("C", "D")})

        pdag = PDAG(ebunch=[("A", "B", "->"), ("D", "C", "->"), ("B", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("D", "C"), ("B", "C"), ("C", "B")})

        pdag = PDAG(ebunch=[("A", "B", "->"), ("D", "C", "->"), ("D", "B", "->"), ("B", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("D", "C"), ("D", "B"), ("B", "C")})

        pdag = PDAG(ebunch=[("A", "B", "->"), ("B", "C", "->"), ("A", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C"), ("A", "C")})

        pdag = PDAG(ebunch=[("A", "B", "->"), ("B", "C", "->"), ("D", "C", "->"), ("A", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(set(cpdag.edges()), {("A", "B"), ("B", "C"), ("A", "C"), ("D", "C")})

        # Examples taken from Perković 2017.
        pdag = PDAG(ebunch=[("V1", "X", "->"), ("X", "V2", "--"), ("V2", "Y", "--"), ("X", "Y", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertEqual(
            set(cpdag.edges()),
            {("V1", "X"), ("X", "V2"), ("X", "Y"), ("V2", "Y"), ("Y", "V2")},
        )

        pdag = PDAG(ebunch=[("Y", "X", "->"), ("V1", "X", "--"), ("X", "V2", "--"), ("V2", "Y", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertEqual(
            set(cpdag.edges()),
            {
                ("X", "V1"),
                ("Y", "X"),
                ("X", "V2"),
                ("V2", "X"),
                ("V2", "Y"),
                ("Y", "V2"),
            },
        )

        # Examples from Bang 2024
        pdag = PDAG(ebunch=[("B", "D", "->"), ("C", "D", "->"), ("A", "D", "--"), ("A", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True, debug=True)
        self.assertEqual(set(cpdag.edges()), {("B", "D"), ("D", "A"), ("C", "A"), ("C", "D")})

        pdag = PDAG(ebunch=[("A", "B", "->"), ("C", "B", "->"), ("D", "B", "--"), ("D", "A", "--"), ("D", "C", "--")])
        cpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(cpdag.edges()),
            {
                ("A", "B"),
                ("C", "B"),
                ("D", "B"),
                ("D", "A"),
                ("A", "D"),
                ("D", "C"),
                ("C", "D"),
            },
        )

        ebunch = [("A", "C", "--"), ("B", "C", "--"), ("D", "C", "--"), ("B", "D", "->"), ("D", "A", "->")]

        pdag = PDAG(ebunch=ebunch)
        mpdag = pdag.apply_meeks_rules(apply_r4=True)
        self.assertSetEqual(
            set(mpdag.edges()),
            {
                ("C", "A"),
                ("C", "B"),
                ("B", "C"),
                ("B", "D"),
                ("D", "A"),
                ("D", "C"),
                ("C", "D"),
            },
        )

        pdag = PDAG(ebunch=ebunch)
        pdag = pdag.apply_meeks_rules()
        self.assertSetEqual(
            set(pdag.edges()),
            {
                ("A", "C"),
                ("C", "A"),
                ("C", "B"),
                ("B", "C"),
                ("B", "D"),
                ("D", "A"),
                ("D", "C"),
                ("C", "D"),
            },
        )

        pdag_inp = PDAG(ebunch=ebunch)
        pdag_inp.apply_meeks_rules(inplace=True)
        self.assertSetEqual(
            set(pdag_inp.edges()),
            {
                ("A", "C"),
                ("C", "A"),
                ("C", "B"),
                ("B", "C"),
                ("B", "D"),
                ("D", "A"),
                ("D", "C"),
                ("C", "D"),
            },
        )

    def test_pdag_equality(self):
        """
        Test the `__eq__` method
        which compares both graph structure and variable-role mappings to allow comparison of two models.
        """
        pdag = PDAG(
            ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )

        # Case1: When the models are the same
        other1 = PDAG(
            ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case2: When the models differ
        other2 = DAG(
            ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "C", "->")],
            latents=["B"],
            roles={"exposures": "A", "adjustment": "D", "outcomes": "C"},
        )
        # Case3: When the directed_ebunch variables differ between models
        other3 = PDAG(
            ebunch=[("A", "C", "->"), ("D", "C", "->"), ("E", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case4: When the directed_ebunch variables differ between models
        other4 = PDAG(
            ebunch=[("A", "E", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case5: When the undirected_ebunch variables differ between models
        other5 = PDAG(
            ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "E", "--")],
            latents=["B"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case6: When the latents variables differ between models
        other6 = PDAG(
            ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["D"],
            roles={"exposures": ("A", "D"), "outcomes": ["C"]},
        )
        # Case7: When the roles variables differ between models
        other7 = PDAG(
            ebunch=[("A", "C", "->"), ("D", "C", "->"), ("B", "A", "--"), ("B", "D", "--")],
            latents=["B"],
            roles={"exposures": ("A"), "adjustment": "D", "outcomes": ["C"]},
        )

        self.assertEqual(pdag.__eq__(other1), True)
        self.assertEqual(pdag.__eq__(other2), False)
        self.assertEqual(pdag.__eq__(other3), False)
        self.assertEqual(pdag.__eq__(other4), False)
        self.assertEqual(pdag.__eq__(other5), False)
        self.assertEqual(pdag.__eq__(other6), False)
        self.assertEqual(pdag.__eq__(other7), False)

    def test_latents_with_role(self):
        self.pdag1 = PDAG(
            ebunch=[
                ("X", "Y", "->"),
                ("A", "B", "--"),
                ("B", "C", "--"),
                ("C", "D", "--"),
                ("D", "E", "--"),
                ("E", "F", "--"),
            ],
            latents=["A"],
            roles={"exposures": "X", "outcomes": "Y", "latents": "B"},
        )
        self.pdag1.with_role(role="latents", variables="C", inplace=True)
        self.pdag1.with_role(role="latents", variables=["D", "E"], inplace=True)
        self.pdag1 = self.pdag1.with_role(role="latents", variables="F", inplace=False)

        self.assertEqual(self.pdag1.latents, {"A", "B", "C", "D", "E", "F"})
        self.assertEqual(set(self.pdag1.get_role("latents")), {"A", "B", "C", "D", "E", "F"})

        with self.assertRaisesRegex(ValueError, "Variable 'G' not found in the graph."):
            self.pdag1.with_role(role="latents", variables="G", inplace=True)

    def test_latnets_without_role(self):
        self.pdag1 = PDAG(
            ebunch=[
                ("X", "Y", "->"),
                ("A", "B", "--"),
                ("B", "C", "--"),
                ("C", "D", "--"),
                ("D", "E", "--"),
                ("E", "F", "--"),
            ],
            latents=["A", "B", "C"],
            roles={"exposures": "X", "outcomes": "Y", "latents": ("D", "E", "F")},
        )

        self.pdag1.without_role(role="latents", variables="A", inplace=True)
        self.pdag1.without_role(role="latents", variables=["B", "C"], inplace=True)
        self.pdag1 = self.pdag1.without_role(role="latents", variables="D", inplace=False)
        self.pdag1 = self.pdag1.without_role(role="latents", variables=["E", "F"], inplace=False)

        self.assertEqual(self.pdag1.latents, set())
        self.assertEqual(set(self.pdag1.get_role("latents")), set())
