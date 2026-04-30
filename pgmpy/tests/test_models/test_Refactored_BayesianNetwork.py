import pytest

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models.Refactored_BayesianNetwork import BayesianNetwork


class TestRefactoredBayesianNetwork:
    def test_model_holds_graph_and_cpd_information(self):
        model = BayesianNetwork(
            ebunch=[("A", "C"), ("B", "C")],
            latents={"L"},
            exposures={"A"},
            outcomes={"C"},
            roles={"instrument": ["B"]},
        )
        model.add_node("L")

        cpd_a = TabularCPD("A", 2, [[0.7], [0.3]])
        cpd_b = TabularCPD("B", 2, [[0.4], [0.6]])
        cpd_c = TabularCPD("C", 2, [[0.9, 0.2, 0.7, 0.1], [0.1, 0.8, 0.3, 0.9]], evidence=["A", "B"], evidence_card=[2, 2])
        cpd_l = TabularCPD("L", 2, [[0.5], [0.5]])

        model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_l)

        assert model.get_cpds("C") == cpd_c
        assert set(model.get_cpds()) == {cpd_a, cpd_b, cpd_c, cpd_l}
        assert model.latents == {"L"}
        assert model.exposures == {"A"}
        assert model.outcomes == {"C"}
        assert model.get_nodes_in_role("instrument") == {"B"}
        assert model.check_model() is True

    def test_check_model_fails_when_cpd_is_missing(self):
        model = BayesianNetwork([("A", "C")])
        model.add_cpds(TabularCPD("A", 2, [[0.5], [0.5]]))

        with pytest.raises(ValueError, match="No CPD associated with C"):
            model.check_model()

    def test_forbidden_responsibility_methods_do_not_exist(self):
        model = BayesianNetwork()

        for method_name in ("fit", "predict", "simulate", "query"):
            assert not hasattr(model, method_name)
