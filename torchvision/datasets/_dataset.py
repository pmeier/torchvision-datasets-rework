import functools
import io
import pathlib
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt

from torch.utils.data import IterDataPipe

from . import decoder as decoder_
from ._builder import DatasetBuilder
from ._home import home
from .decoder import _decode_sample

__all__ = ["Dataset"]


class Dataset:
    def __init__(
        self, builder: DatasetBuilder, *, data_dir: Optional[pathlib.Path] = None, **config_options: Any
    ) -> None:
        self._builder = builder
        self.config = builder._make_config(**config_options)

        if data_dir is None:
            data_dir = home()
        elif isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)
        data_dir /= self.name.lower()
        self.data_dir = data_dir

    @property
    def name(self) -> str:
        return self._builder.NAME

    @property
    def meta(self):
        return self._builder.META

    def label_to_class(self, label: int) -> str:
        return self._builder.label_to_class(label)

    def check_integrity(self, *, strict: bool = True) -> bool:
        # TODO: async
        return all(
            resource.check_integrity(self.data_dir, strict=strict)
            for resource in self._builder.resources(self.config).values()
        )

    def download(self, *, strict: bool = True):
        # TODO: async
        for resource in self._builder.resources(self.config).values():
            resource.download(self.data_dir, strict=strict)

    def as_datapipe(
        self, *, decoder: Optional[Callable[[io.BufferedIOBase], Any]] = None
    ) -> IterDataPipe[Dict[str, Any]]:
        datapipe = self._builder.datapipe(self.config, data_dir=self.data_dir)
        if decoder:
            datapipe = datapipe.map(functools.partial(_decode_sample, decoder=decoder))
        return datapipe

    def show_example(self):
        sample = next(iter(self.as_datapipe(decoder=decoder_.pil)))
        plt.imshow(sample["image"])
        plt.title(f"{sample['cls']} ({sample['label']})")
        plt.show()
