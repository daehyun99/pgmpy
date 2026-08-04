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
    _CONVERTERS = {
        "normal": _convert_normal,
    }

    def __init__(self, posterior_type: str = "normal"):
        self.posterior_type = posterior_type

    def convert(self, samples, index, columns, name="obs"):
        converter = self._CONVERTERS.get(self.posterior_type)

        if converter is None:
            raise ValueError(f"{self.posterior_type} is not ...")

        return converter(samples, index, columns, name=name)
