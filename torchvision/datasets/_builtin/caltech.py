import pathlib
import re
from typing import Any, Dict, Iterable, List, Set, Tuple

import torch
from torch.utils.data import datapipes as dp
from torch.utils.data.datapipes.utils.decoder import mathandler
from torchvision.datasets._dataset import Dataset
from torchvision.datasets.utils import Resource

__all__ = ["Caltech101", "Caltech256"]


class Caltech101(Dataset):
    _ANNOTATIONS_CLASS_MAP = {
        "Faces_2": "Faces",
        "Faces_3": "Faces_easy",
        "Motorbikes_16": "Motorbikes",
        "Airplanes_Side_2": "airplanes",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._annotations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._cls_to_label: Dict[str, int] = {}

    def resources(self) -> List[Resource]:
        return [
            Resource(
                "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
                sha512="70745746dac6a2f38a906314272aee0b45f6a5a12a1fa34d01fbc9799c020eab749dd3722aaea75c662545c356fb46e8012674645e442a5d30ce0f9479f9385b",
            ),
            Resource(
                "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
                sha512="365ecca3f8604c2b83b1546377dbc85bf1840234898e5f1543e4b0d54e19f6432bf4f610f5d52c9b37577b0c3166d52239cbb8b2c515c186540dd83d9e344d02",
            ),
        ]

    def _load_annotations(self) -> None:
        datapipe: Iterable = (str(self.data_dir / "Annotations.tar"),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromTar(datapipe)
        datapipe = dp.iter.RoutedDecoder(datapipe, mathandler())

        annotations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        classes: Set[str] = set()
        name_pattern = re.compile(r"annotation_(?P<id>\d+)[.]mat")
        for path, annotation in datapipe:
            path = pathlib.Path(path)
            if not name_pattern.match(path.name):
                continue

            cls = path.parent.name
            if cls in self._ANNOTATIONS_CLASS_MAP:
                cls = self._ANNOTATIONS_CLASS_MAP[cls]
            classes.add(cls)
            image_name = name_pattern.sub(r"image_\g<id>.jpg", path.name)
            key = (cls, image_name)

            val = dict(annotation_path=path, contour=torch.as_tensor(annotation["obj_contour"]))

            annotations[key] = val

        self._annotations = annotations
        self._cls_to_label = {cls: label for label, cls in enumerate(sorted(classes))}

    def _remove_rogue_images(self, data: Tuple[str, Any]) -> bool:
        path = pathlib.Path(data[0])

        return path.parent.name != "BACKGROUND_Google" or path.name == "RENAME2"

    def _collate_sample(self, data: Tuple[str, Any]) -> Dict[str, Any]:
        path, image = data
        path = pathlib.Path(path)
        cls = path.parent.name
        label = self._cls_to_label[cls]
        sample = dict(image=image, image_path=path, label=label, cls=cls)
        sample.update(self._annotations[(cls, path.name)])
        return sample

    @property
    def dp(self) -> Iterable[Dict[str, Any]]:
        if not self._annotations:
            self._load_annotations()

        datapipe: Iterable = (str(self.data_dir / "101_ObjectCategories.tar.gz"),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromTar(datapipe)
        datapipe = dp.iter.Filter(datapipe, self._remove_background_images)
        return dp.iter.Map(datapipe, self._collate_sample)


class Caltech256(Dataset):
    def resources(self) -> List[Resource]:
        return [
            Resource(
                "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
                md5="67b4f42ca05d46448c6bb8ecd2220f6d",
                sha512="c71e604fa72092f5235dfdde8821ae58d3c7ba35ee0ce81bfbf397e1061f6f89f84c7ee400049244a3bd940b63a45a3754d7a788c5d2cb3f60baee02385c665a",
            ),
        ]

    def _collate_sample(self, data: Tuple[str, Any]) -> Dict[str, Any]:
        path, image = data
        path = pathlib.Path(path)
        label_, cls = path.parent.name.split(".")
        label = int(label_)
        return dict(image=image, path=path, label=label, cls=cls)

    @property
    def dp(self) -> Iterable[Dict[str, Any]]:
        datapipe: Iterable = (str(self.data_dir / "256_ObjectCategories.tar"),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromTar(datapipe)
        return dp.iter.Map(datapipe, self._collate_sample)
