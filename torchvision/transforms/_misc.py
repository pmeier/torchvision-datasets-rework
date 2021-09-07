from typing import Any, Dict, Sequence

import torch

from torchvision.features import Image
from torchvision.transforms import Transform

__all__ = ["Normalize"]


class Normalize(Transform):
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.mean = mean
        self.std = std

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(mean=self.mean, std=self.std)

    @staticmethod
    def _channel_stats_to_tensor(stats: Sequence[float], *, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(stats, device=like.device, dtype=like.dtype).view(-1, 1, 1)

    @staticmethod
    def image(input: Image, *, mean: Sequence[float], std: Sequence[float]) -> Image:
        mean_t = Normalize._channel_stats_to_tensor(mean, like=input)
        std_t = Normalize._channel_stats_to_tensor(std, like=input)
        return (input - mean_t) / std_t

    def extra_repr(self) -> str:
        return f"mean={tuple(self.mean)}, std={tuple(self.std)}"
