import contextlib
import difflib
import functools
import io
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt

from torch.utils.data import IterDataPipe

from torchvision.datasets import Dataset, DatasetInfo
from torchvision.datasets.decoder import decode_sample, pil
from torchvision.datasets.utils import Sample

from . import _builtin
from ._home import home

__all__ = ["home", "register", "list", "load", "show_example"]


_DATASETS: Dict[str, Dataset] = {}


def register(dataset: Dataset) -> None:
    _DATASETS[dataset.name] = dataset


for name, obj in _builtin.__dict__.items():
    if name.startswith("_"):
        continue

    with contextlib.suppress(TypeError):
        if issubclass(obj, Dataset) and obj is not Dataset:
            register(obj())


def list() -> List[str]:
    return sorted(_DATASETS.keys())


def _find(name: str) -> Dataset:
    name = name.lower()
    with contextlib.suppress(KeyError):
        return _DATASETS[name]

    msg = f"Unknown dataset '{name}'."
    close_matches = difflib.get_close_matches(name, _DATASETS.keys(), n=1)
    if close_matches:
        msg += f" Did you mean '{close_matches[0]}'?"
    raise ValueError(msg)


def info(name: str) -> DatasetInfo:
    return _find(name).info


def load(
    name: str,
    *,
    root: Optional[Union[str, pathlib.Path]] = None,
    decoder: Optional[Callable[[io.BufferedIOBase], Any]] = pil,
    split: str = "train",
    **options: Any,
) -> IterDataPipe[Sample]:
    dataset = _find(name)
    config = dataset.info.make_config(split=split, **options)

    if root is None:
        root = home() / dataset.name
    elif isinstance(root, str):
        root = pathlib.Path(root)

    dataset.download(root, config=config)

    datapipe = dataset.datapipe(config, data_dir=root)
    if decoder:
        datapipe = datapipe.map(functools.partial(decode_sample, decoder=decoder))

    return datapipe


def show_example(dataset: Union[str, Iterable[Sample]]):
    if isinstance(dataset, str):
        dataset = _find(dataset)

    sample = next(iter(dataset))
    plt.imshow(sample.image)

    if hasattr(sample, "cls"):
        title = f"{sample.cls} ({sample.label})"
    elif hasattr(sample, "label"):
        title = str(sample.label)
    else:
        title = None
    if title:
        plt.title(title)

    plt.show()
