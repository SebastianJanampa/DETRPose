from torch import nn
from omegaconf import OmegaConf
from huggingface_hub import PyTorchModelHubMixin
from detrpose.core import instantiate


class HFModel(
    nn.Module,
    PyTorchModelHubMixin,
    license="apache-2.0",
):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = OmegaConf.create(config)
        self.model = instantiate(cfg['model'])
        self.postprocessor = instantiate(cfg['postprocessor'])

    def deploy(self):
        self.eval()
        self.model = self.model.deploy()
        self.postprocessor = self.postprocessor.deploy()
        return self

    def forward(self, samples, orig_target_sizes):
        out = self.model(samples)
        res = self.postprocessor(out, orig_target_sizes)
        return res

