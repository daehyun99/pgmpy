from skpro.distributions.normal import Normal as SkproNormal


class PyroToSkpro:
    _CONVERTERS = {
        "normal": "_convert_normal",
    }

    def __init__(self, posterior_type: str, device):
        self.posterior_type = posterior_type
        self.device = device

    def convert(self, samples, index, columns, name="obs"):
        converter_name = self._CONVERTERS.get(self.posterior_type)

        if converter_name is None:
            raise ValueError(f"{self.posterior_type} is not ...")

        converter = getattr(self, converter_name)

        return converter(samples, index, columns, name=name)

    def _convert_normal(self, samples, index, columns, name="obs"):
        y_samples = samples[name]
        if self.device == "cuda":
            mean = y_samples.mean(dim=0).detach().reshape(-1, 1).cpu().numpy()
            sigma = y_samples.std(dim=0, unbiased=False).detach().reshape(-1, 1).cpu().numpy()
        else:
            mean = y_samples.mean(dim=0).detach().reshape(-1, 1).numpy()
            sigma = y_samples.std(dim=0, unbiased=False).detach().reshape(-1, 1).numpy()

        return SkproNormal(
            mu=mean,
            sigma=sigma,
            index=index,
            columns=columns,
        )
