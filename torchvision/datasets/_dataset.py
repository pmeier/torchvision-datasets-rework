import abc
import csv
import io
import pathlib
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from torch.utils.data import IterDataPipe

from torchvision.datasets.utils import Config, Resource, Sample

__all__ = ["DatasetInfo", "Dataset"]


class DatasetInfo:
    def __init__(
        self,
        name: str,
        *,
        citation: Optional[str] = None,
        homepage: Optional[str] = None,
        license: Optional[str] = None,
        num_samples: Optional[int] = None,
        classes: Optional[pathlib.Path] = None,
        options: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> None:
        self.name = name.lower()
        if isinstance(classes, pathlib.Path):
            with open(classes, newline="") as fh:
                classes = {int(row["label"]): row["class"] for row in csv.DictReader(fh)}
        self.classes = classes
        if classes:
            self.labels = {cls: label for label, cls in classes.items()}

        if options is None:
            options = dict(split=("train",))
        self.options = options

    @property
    def default_config(self) -> Config:
        return Config({name: valid_args[0] for name, valid_args in self.options.items()})

    def make_config(self, **options: Any) -> Config:
        for name, arg in options.items():
            if name not in self.options:
                raise ValueError

            valid_args = self.options[name]

            if arg not in valid_args:
                raise ValueError

        return Config(self.default_config, **options)

    def label_to_class(self, label: int) -> str:
        # TODO check if classes is available
        return self.classes[label]

    def class_to_label(self, cls: str) -> int:
        # TODO check if classes is available
        return self.labels[cls]


class Dataset(abc.ABC):
    @property
    @abc.abstractmethod
    def info(self) -> DatasetInfo:
        pass

    @abc.abstractmethod
    def resources(self, config: Config) -> Dict[str, Resource]:
        pass

    @abc.abstractmethod
    def datapipe(self, config: Config, *, data_dir: pathlib.Path) -> IterDataPipe[Dict[str, Any]]:
        pass

    @property
    def name(self) -> str:
        return self.info.name

    def _resource_path(
        self, root: Union[str, pathlib.Path], *, key: str, config: Optional[Config] = None
    ) -> pathlib.Path:
        return pathlib.Path(root) / self.resources(config or self.info.default_config)[key].name

    def check_integrity(self, *, strict: bool = True) -> bool:
        # TODO: async
        return all(
            resource.check_integrity(self.data_dir, strict=strict)
            for resource in self._builder.resources(self.config).values()
        )

    def download(self, root: Union[str, pathlib.Path], *, config: Optional[Config] = None, strict: bool = True):
        # TODO: async
        for resource in self.resources(config or self.info.default_config).values():
            resource.download(root, strict=strict)

    @staticmethod
    def default_collate_sample(
        data: Tuple[str, io.BufferedIOBase],
        annotations: Optional[Callable[[pathlib.Path], Dict[str, Any]]] = None,
    ) -> Sample:
        path, image = data
        path = pathlib.Path(path)
        sample = Sample(image=image, image_path=path)

        if not annotations:
            return sample

        sample.update(annotations(path))
        return sample
