from pgmpy.parameterization._base import BaseParameter


class TestBaseParameter:
    """Test BaseParameter class"""

    def test_base_parameter_default(self):
        parameter = BaseParameter()

        assert parameter.name == "BaseParameter"
        assert parameter.get_class_tag("variable_type") == "discrete"
        assert parameter.get_class_tag("produces_factor") is False
        assert parameter.get_class_tag("is_linear_gaussian") is False
        assert parameter.get_class_tag("supports_fit_joint") is False
        assert parameter.get_class_tag("python_dependencies") == ()
