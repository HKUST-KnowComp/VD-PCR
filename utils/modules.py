from collections import Counter, defaultdict
import logging
from typing import Union, List, Dict, Any
import torch
from torch import nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Reshaper(nn.Module):
    def __init__(self, *output_shape):
        super().__init__()

        self.output_shape = output_shape

    def forward(self, input: torch.Tensor):
        return input.view(*self.output_shape)


class Normalizer(nn.Module):
    def __init__(self, target_norm=1.):
        super().__init__()
        self.target_norm = target_norm

    def forward(self, input: torch.Tensor):
        return input * self.target_norm / input.norm(p=2, dim=1, keepdim=True)


class Squeezer(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return torch.squeeze(input, dim=self.dim)
