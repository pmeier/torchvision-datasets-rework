import contextlib
import difflib
import pathlib
from typing import Any, Dict, List, Optional, Union

from . import _builtin
from ._builder import DatasetBuilder
from ._dataset import Dataset
from ._home import home

__all__ = ["home", "register", "list", "get"]


_BUILDERS: Dict[str, DatasetBuilder] = {}


def register(builder: DatasetBuilder) -> None:
    _BUILDERS[builder.NAME] = builder


for name, obj in _builtin.__dict__.items():
    if name.startswith("_"):
        continue

    with contextlib.suppress(TypeError):
        if issubclass(obj, DatasetBuilder) and obj is not DatasetBuilder:
            register(obj())


def list() -> List[str]:
    return sorted(_BUILDERS.keys())


def _find(name: str) -> DatasetBuilder:
    name = name.lower()
    with contextlib.suppress(KeyError):
        return _BUILDERS[name]

    msg = f"Unknown dataset '{name}'."
    close_matches = difflib.get_close_matches(name, _BUILDERS.keys(), n=1)
    if close_matches:
        msg += f" Did you mean '{close_matches[0]}'?"
    raise ValueError(msg)


def show():
    builder = _find(name)
    return repr(builder)


def get(
    name: str,
    *,
    data_dir: Optional[Union[str, pathlib.Path]] = None,
    download: bool = True,
    split: str = "train",
    **options: Any,
) -> Dataset:
    builder = _find(name)
    dataset = Dataset(builder, data_dir=data_dir, split=split, **options)

    if download:
        dataset.download(strict=False)
    elif not dataset.check_integrity():
        raise RuntimeError()

    return dataset
