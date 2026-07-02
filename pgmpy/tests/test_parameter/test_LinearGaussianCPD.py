import pytest

from pgmpy.parameter.LinearGaussianCPD import LinearGaussianCPD


@pytest.fixture
def continue_data(): ...


class TestLinearGaussianCPD:
    def test_base_parameter_default(self):
        parameter = LinearGaussianCPD()

    def test_fit(self, continue_data): ...

    def test_predict_proba(self, continue_data): ...

    def test_set_values(self): ...
