import io
import pathlib
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from torch.utils.data import IterDataPipe

from .utils import Resource

__all__ = ["Dataset"]


class Dataset(IterDataPipe):
    def __init__(self, data_dir: pathlib.Path, *, split: str, **kwargs) -> None:
        if kwargs:
            raise ValueError()

        self.data_dir = data_dir
        self.split = split

    def resources(self) -> List[Resource]:
        raise NotImplementedError

    @property
    def dp(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError

    def check_integrity(self, *, strict: bool = True) -> bool:
        return all(resource.check_integrity(self.data_dir, strict=strict) for resource in self.resources())

    def download(self):
        for resource in self.resources():
            resource.download(self.data_dir)

    @staticmethod
    def _default_collate_sample(
        data: Tuple[str, io.BufferedIOBase],
        annotations: Optional[Dict[str, Dict[str, Any]]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        path, image = data
        path = pathlib.Path(path)
        sample = dict(image=image, image_path=path)

        if not annotations:
            return sample

        try:
            annotation = annotations[path.name]
        except KeyError:
            if strict:
                raise
            else:
                return sample

        sample.update(annotation)
        return sample

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self.dp

    def __len__(self):
        raise NotImplemented
