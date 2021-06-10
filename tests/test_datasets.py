import contextlib

import pytest

import torchvision.datasets._builtin as builtin_datasets
from torchvision import datasets


class TestApi:
    def test_list(self):
        builtin_datasets_names = set()
        for name, obj in builtin_datasets.__dict__.items():
            if name.startswith("_"):
                continue

            with contextlib.suppress(TypeError):
                if issubclass(obj, datasets.Dataset) and obj is not datasets.Dataset:
                    builtin_datasets_names.add(name.lower())

        assert set(datasets.list()) == builtin_datasets_names


class TestDatasets:
    def test_instantiation(self):
        pass

    def test_consume(self):
        pass
