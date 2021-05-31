import difflib
import functools
import io
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Type, Union

from torch.hub import _get_torch_home

from . import _builtin
from ._dataset import Dataset
from .decoder import _decode_sample, pil

__all__ = ["home", "register", "list", "get"]


_HOME = pathlib.Path(_get_torch_home()) / "torchvision" / "datasets"


def home(home: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    global _HOME
    if home is not None:
        _HOME = pathlib.Path(home).expanduser().resolve()
        return _HOME

    home = os.getenv("TORCHVISION_DATASETS_HOME")
    if home is not None:
        return pathlib.Path(home)

    return _HOME


_DATASET_CLASSES: Dict[str, Type[Dataset]] = {}


def register(name: str, cls: Type[Dataset]) -> None:
    _DATASET_CLASSES[name.lower()] = cls


for name, obj in _builtin.__dict__.items():
    if name.startswith("_"):
        continue
    if issubclass(obj, Dataset) and obj is not Dataset:
        register(name, obj)


def list() -> List[str]:
    return sorted(_DATASET_CLASSES.keys())


def show():
    pass


def get(
    name: str,
    *,
    data_dir: Optional[Union[str, pathlib.Path]] = None,
    download: bool = True,
    split: str = "train",
    decoder: Optional[Callable[[io.BufferedIOBase], Any]] = pil,
    **dataset_kwargs: Any,
) -> Dataset:
    name = name.lower()

    try:
        dataset_cls = _DATASET_CLASSES[name]
    except KeyError as error:
        msg = f"Unknown dataset '{name}'."
        close_matches = difflib.get_close_matches(name, _DATASET_CLASSES.keys(), n=1)
        if close_matches:
            msg += f" Did you mean '{close_matches[0]}'?"
        raise ValueError(msg) from error

    if data_dir is None:
        data_dir = home()
    elif isinstance(data_dir, str):
        data_dir = pathlib.Path(data_dir)
    data_dir /= name

    dataset = dataset_cls(data_dir=data_dir, split=split, **dataset_kwargs)
    if download:
        dataset.download()
    elif not dataset.check_integrity():
        raise RuntimeError()

    if decoder:
        dataset = dataset.map(functools.partial(_decode_sample, decoder=decoder))

    return dataset
