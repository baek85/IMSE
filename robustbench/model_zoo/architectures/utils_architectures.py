import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple
from torch import Tensor


class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float],
        std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std


def normalize_model(model: nn.Module, mean: Tuple[float, float, float],
    std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return nn.Sequential(layers)

def normalize_timm_model(model):
    print("Using timm model with mean {} and std {}".format(
        model.default_cfg['mean'], model.default_cfg['std']))  # type: ignore
    return normalize_model(
        model,
        mean=model.default_cfg['mean'],  # type: ignore
        std=model.default_cfg['std'])  # type: ignore