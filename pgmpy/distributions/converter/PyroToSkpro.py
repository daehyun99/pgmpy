from skpro.distributions.normal import Normal as SkproNormal


def _convert_normal(samples, index, columns, name="obs"):
    y_samples = samples[name]
    mean = y_samples.mean(dim=0).detach().reshape(-1, 1).cpu().numpy()
    sigma = y_samples.std(dim=0, unbiased=False).detach().reshape(-1, 1).cpu().numpy()

    return SkproNormal(
        mu=mean,
        sigma=sigma,
        index=index,
        columns=columns,
    )


class PyroToSkpro:
    _DEFAULT_CONVERTERS = {
        "normal": _convert_normal,
    }

    def __init__(self, posterior_type: str = "normal"):
        self.posterior_type = posterior_type

        self._converters = self._DEFAULT_CONVERTERS

    def convert(self, samples, index, columns, name="obs"):
        converter = self._converters.get(self.posterior_type)

        if converter is None:
            raise ValueError(f"{self.posterior_type} is not ...")

        return converter(samples, index, columns, name=name)

    def add(
        self,
        posterior_type,
        converter_fn,
    ):
        if not callable(converter_fn):
            raise TypeError

        self._converters[posterior_type] = converter_fn
        return self

    def set_converter(self, posterior_type):
        self.posterior_type = posterior_type
        return self

    def remove(
        self,
        posterior_type,
    ):
        self._converters.pop(posterior_type)
        return self

    def get(self):
        """
        Returns a list of all available conversion methods.
        """
        return list(self._converters.keys())
