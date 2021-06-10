import os
import pathlib
from typing import Optional, Union

from torch.hub import _get_torch_home

# This needs a better default
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
