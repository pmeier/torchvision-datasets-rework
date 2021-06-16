import functools
import pathlib
import re
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import IterDataPipe
from torch.utils.data import datapipes as dp
from torch.utils.data.datapipes.utils.decoder import mathandler

from torchvision.datasets import Dataset, DatasetInfo
from torchvision.datasets.utils import Config, Resource, Sample


class Caltech101(Dataset):
    _ANNOTATIONS_CLASS_MAP = {
        "Faces_2": "Faces",
        "Faces_3": "Faces_easy",
        "Motorbikes_16": "Motorbikes",
        "Airplanes_Side_2": "airplanes",
    }

    def __init__(self):
        self._annotations_map: Optional[Dict] = None

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            "caltech101",
            homepage="http://www.vision.caltech.edu/Image_Datasets/Caltech101/",
            classes=pathlib.Path(__file__).parent / "caltech101.csv",
        )

    def resources(self, config: Config) -> Dict[str, Resource]:
        return dict(
            images=Resource(
                "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
                sha512="365ecca3f8604c2b83b1546377dbc85bf1840234898e5f1543e4b0d54e19f6432bf4f610f5d52c9b37577b0c3166d52239cbb8b2c515c186540dd83d9e344d02",
            ),
            annotations=Resource(
                "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
                sha512="70745746dac6a2f38a906314272aee0b45f6a5a12a1fa34d01fbc9799c020eab749dd3722aaea75c662545c356fb46e8012674645e442a5d30ce0f9479f9385b",
            ),
        )

    def _remove_background_images(self, data: Tuple[str, Any]) -> bool:
        return pathlib.Path(data[0]).parent.name != "BACKGROUND_Google"

    def _remove_rogue_image(self, data: Tuple[str, Any]) -> bool:
        return pathlib.Path(data[0]).name != "RENAME2"

    @functools.lru_cache(maxsize=1)
    def _load_annotations_map(self, data_dir, *, config) -> Dict[Tuple[str, str], Dict[str, Any]]:
        annotations_archive = self._resource_path(data_dir, config=config, key="annotations")
        datapipe: Iterable = (str(annotations_archive),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromTar(datapipe)
        datapipe = dp.iter.RoutedDecoder(datapipe, mathandler())

        annotations_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        name_pattern = re.compile(r"annotation_(?P<id>\d+)[.]mat")
        for path, annotation in datapipe:
            path = pathlib.Path(path)
            if not name_pattern.match(path.name):
                continue

            cls = path.parent.name
            if cls in self._ANNOTATIONS_CLASS_MAP:
                cls = self._ANNOTATIONS_CLASS_MAP[cls]
            image_name = name_pattern.sub(r"image_\g<id>.jpg", path.name)
            key = (cls, image_name)

            val = dict(
                annotation_path=path,
                contour=torch.as_tensor(annotation["obj_contour"]),
                cls=cls,
                label=self.info.class_to_label(cls),
            )

            annotations_map[key] = val

        self._annotations_map = annotations_map
        return annotations_map

    def _annotations(
        self, path: pathlib.Path, *, annotations_map: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[str, Any]:
        cls = path.parent.name
        image_name = path.name
        return annotations_map[(cls, image_name)]

    def datapipe(self, config, *, data_dir: pathlib.Path) -> IterDataPipe[Sample]:
        images_archive = self._resource_path(data_dir, config=config, key="images")
        datapipe: Iterable = (str(images_archive),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromTar(datapipe)
        datapipe = dp.iter.Filter(datapipe, self._remove_background_images)
        datapipe = dp.iter.Filter(datapipe, self._remove_rogue_image)
        return dp.iter.Map(
            datapipe,
            functools.partial(
                self.default_collate_sample,
                annotations=lambda path: self._annotations(
                    path, annotations_map=self._load_annotations_map(data_dir, config=config)
                ),
            ),
        )


class Caltech256(Dataset):
    @property
    def info(self):
        return DatasetInfo(
            "caltech256",
            homepage="http://www.vision.caltech.edu/Image_Datasets/Caltech256/",
            num_samples=30607,
            classes=pathlib.Path(__file__).parent / "caltech256.csv",
        )

    def resources(self, config: Config) -> Dict[str, Resource]:
        return dict(
            images=Resource(
                "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
                md5="67b4f42ca05d46448c6bb8ecd2220f6d",
                sha512="c71e604fa72092f5235dfdde8821ae58d3c7ba35ee0ce81bfbf397e1061f6f89f84c7ee400049244a3bd940b63a45a3754d7a788c5d2cb3f60baee02385c665a",
            ),
        )

    def _annotations(self, path: pathlib.Path) -> Dict[str, Any]:
        label = int(path.parent.name.split(".")[0])
        return dict(label=label, cls=self.info.label_to_class(label))

    def datapipe(self, config, *, data_dir: pathlib.Path) -> IterDataPipe[Sample]:
        images_archive = self._resource_path(data_dir, config=config, key="images")
        datapipe: Iterable = (str(images_archive),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromTar(datapipe)
        return dp.iter.Map(
            datapipe,
            functools.partial(self.default_collate_sample, annotations=self._annotations),
        )
