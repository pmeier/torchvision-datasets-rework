import functools
import hashlib
import pathlib
import urllib.request
from typing import Optional, Union

from torch.hub import tqdm

import torchvision

__all__ = ["Resource"]


class Resource:
    _CHUNK_SIZE = 1024 * 1024 * 50
    _USER_AGENT = f"torchvision=={torchvision.__version__}"

    def __init__(self, url: str, **checksums: Optional[str]) -> None:
        self.url = url
        for type, checksum in checksums.items():
            if not hasattr(hashlib, type):
                raise ValueError()

            setattr(self, type, checksum)
        self.checksums = checksums

    @property
    def name(self) -> str:
        return pathlib.Path(self.url).name

    @staticmethod
    @functools.lru_cache()
    def _compute_checksum(file: pathlib.Path, type: str, *, chunk_size: int = _CHUNK_SIZE) -> str:
        hash = getattr(hashlib, type)()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash.update(chunk)
        return hash.hexdigest()

    def _make_file(self, dir: Union[str, pathlib.Path], name: Optional[str]) -> pathlib.Path:
        return pathlib.Path(dir).resolve() / (name or self.name)

    def check_integrity(
        self,
        dir: Union[str, pathlib.Path],
        *,
        name: Optional[str] = None,
        strict: bool = True,
    ) -> bool:
        file = self._make_file(dir, name)

        if not file.exists():
            return False

        if strict and not self.checksums:
            return False

        return all(self._compute_checksum(file, type) == checksum for type, checksum in self.checksums.items())

    def download(
        self,
        dir: Union[str, pathlib.Path],
        *,
        name: Optional[str] = None,
        strict: bool = True,
        chunk_size: int = _CHUNK_SIZE,
    ) -> None:
        if self.check_integrity(dir, name=name, strict=strict):
            return

        file = self._make_file(dir, name)
        file.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        with open(file, "wb") as fh:
            with urllib.request.urlopen(
                urllib.request.Request(self.url, headers={"User-Agent": self._USER_AGENT})
            ) as response:
                with tqdm(total=response.length) as pbar:
                    for chunk in iter(lambda: response.read(chunk_size), ""):
                        if not chunk:
                            break
                        pbar.update(chunk_size)
                        fh.write(chunk)

        if self.check_integrity(dir, name=name, strict=strict):
            raise RuntimeError()

    def __repr__(self) -> str:
        head = f"{type(self).__name__}("
        body = [f"url={self.url}"]
        if self.checksums:
            for type_, checksum in sorted(self.checksums.items()):
                body.append(f"{type_}={checksum}")
        tail = ")"
        return "\n".join([head] + [f"\t{line}" for line in body] + [tail])
