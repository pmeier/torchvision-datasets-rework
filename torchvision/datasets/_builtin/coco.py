import functools
import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from torch.utils.data import IterDataPipe
from torch.utils.data import datapipes as dp

from torchvision.datasets import Dataset, DatasetInfo
from torchvision.datasets.utils import Config, Resource, Sample

__all__ = ["Coco"]


class Coco(Dataset):
    @property
    def info(self):
        return DatasetInfo(
            "coco",
            classes=pathlib.Path(__file__).parent / "coco.csv",
            options=dict(
                split=("train", "val", "test"),
                year=("2014",),
            ),
        )

    def resources(self, config) -> Dict[str, Resource]:
        if config.year == "2014":
            if config.split in ("train", "val"):
                if config.split == "train":
                    images = Resource(
                        "http://images.cocodataset.org/zips/train2014.zip",
                        sha512="7e7c25dbc992008f8d55e06c5bd31d96bbb68f68f944e6030dffd7f2b1158d6601f25aa20038a5a2041792430e4de1361e50066d50072918246982451c740949",
                    )
                else:  # self.split == "val":
                    images = Resource()
                annotations = Resource(
                    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                    sha512="e8b235333392e9dc7e2ef06b831e69fcc74d39f8acd617276ed813d5c28b7e278856aefce9a0c2210df87c5f8f491857e8fa90591bd441ebcf82f05e14fe5cc0",
                )
            else:  # self.split == "test"
                images = Resource()
                annotations = Resource()
        else:
            raise RuntimeError

        return dict(images=images, annotations=annotations)

    @functools.lru_cache(maxsize=1)
    def _load_annotations_map(self, data_dir, *, config) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        annotations_archive = self._resource_path(data_dir, config=config, key="annotations")
        datapipe: Iterable = (str(annotations_archive),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromZip(datapipe)

        path, fh = next(iter(datapipe))
        foo = json.load(fh)
        fh.close()

        path = pathlib.Path(path)
        annotations = defaultdict(lambda: {})
        image_id_to_name = {meta["id"]: meta["file_name"] for meta in foo["images"]}
        for meta in foo["annotations"]:
            image_name = image_id_to_name[meta["image_id"]]
            annotation_id = meta["annotation_id"] = meta.pop("id")
            meta["annotation_path"] = path
            annotations[image_name][annotation_id] = meta

        # Bring the annotations in the default format
        # Dict[str, Dict[int, Dict[str, Any]]] -> Dict[str, Dict[str, List[Dict[str, Any]]]]
        annotations_map = {
            image_name: dict(annotations=list(tuple(zip(*sorted(tuple(annotation.items()))))[1]))
            for image_name, annotation in annotations.items()
        }
        # Some images don't have annotations
        annotations_map.update(
            {
                image_name: dict(annotations=[])
                for image_name in set(image_id_to_name.values()) - set(annotations.keys())
            }
        )
        return annotations_map

    def datapipe(self, config: Config, *, data_dir: pathlib.Path) -> IterDataPipe[Sample]:
        images_archive = self._resource_path(data_dir, config=config, key="images")
        datapipe: Iterable = (str(images_archive),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromZip(datapipe)
        return dp.iter.Map(
            datapipe,
            functools.partial(
                self.default_collate_sample,
                annotations=lambda path: self._load_annotations_map(data_dir, config=config)[path.name],
            ),
        )
