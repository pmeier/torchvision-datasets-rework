import abc
import io
import pathlib
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from torch.utils.data import IterDataPipe
from torchvision.datasets.utils import Resource

from ._meta import Meta


class DatasetBuilder(abc.ABC):
    NAME: str = None
    META: Meta = None
    CONFIG_OPTIONS: Dict[str, Sequence[Any]] = None

    def __init__(self):
        if self.NAME is None:
            self.NAME = type(self).__name__.lower().replace("builder", "")
        if self.META is None:
            self.META = Meta()
        if self.CONFIG_OPTIONS is None:
            self.CONFIG_OPTIONS = dict(split=("train",))

    @abc.abstractmethod
    def resources(self, config: SimpleNamespace) -> Dict[str, Resource]:
        pass

    @abc.abstractmethod
    def datapipe(self, config: SimpleNamespace, *, data_dir: pathlib.Path) -> IterDataPipe[Dict[str, Any]]:
        pass

    def label_to_class(self, label: int) -> str:
        # TODO check if classes is available
        return self.META.classes[label]

    def class_to_label(self, cls: str) -> int:
        return self.META.labels[cls]

    @property
    def _default_config(self) -> Dict[str, Any]:
        return {name: valid_args[0] for name, valid_args in self.CONFIG_OPTIONS.items()}

    def _make_config(self, **options: Any):
        for name, arg in options.items():
            if name not in self.CONFIG_OPTIONS:
                raise ValueError

            valid_args = self.CONFIG_OPTIONS[name]

            if arg not in valid_args:
                raise ValueError

        return SimpleNamespace(**dict(self._default_config, **options))

    def _resource_path(
        self, data_dir: Union[str, pathlib.Path], *, key: str, config: Optional[Any] = None
    ) -> pathlib.Path:
        return pathlib.Path(data_dir) / self.resources(config or self._default_config)[key].name

    @staticmethod
    def _default_collate_sample(
        data: Tuple[str, io.BufferedIOBase],
        annotations: Optional[Callable[[pathlib.Path], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        path, image = data
        path = pathlib.Path(path)
        sample = dict(image=image, image_path=path)

        if not annotations:
            return sample

        sample.update(annotations(path))
        return sample
