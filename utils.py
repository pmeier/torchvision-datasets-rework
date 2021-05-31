import contextlib
import pathlib
import unittest.mock

from torchvision import datasets

__all__ = ["use_dataset_mocks"]


@contextlib.contextmanager
def use_dataset_mocks():
    old_home = datasets.home()
    here = pathlib.Path(__file__).parent
    home = here / "datasets"

    files = {path.name: path for path in home.glob("**/*") if path.is_file()}

    unpatched_init = datasets.utils.Resource.__init__

    def patched_init(self, url, **checksums):
        name = pathlib.Path(url).name
        nonlocal files
        try:
            file = files[name]
        except KeyError:
            return unpatched_init(self, url, **checksums)

        patched_checksums = {type: self._compute_checksum(file, type) for type in checksums.keys()}
        return unpatched_init(self, url, **patched_checksums)

    with unittest.mock.patch("torchvision.datasets.utils.Resource.__init__", new=patched_init):
        datasets.home(home)
        try:
            yield
        finally:
            datasets.home(old_home)
