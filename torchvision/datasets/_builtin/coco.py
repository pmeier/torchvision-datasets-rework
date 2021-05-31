import functools
import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from torch.utils.data import datapipes as dp
from torchvision.datasets._dataset import Dataset
from torchvision.datasets.utils import Resource

__all__ = ["Coco"]


class Coco(Dataset):
    def __init__(self, *args: Any, year: str = "2014", **kwargs: Any) -> None:
        self.year = year
        super().__init__(*args, **kwargs)
        self._annotations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._missing_keys = set()

    def resources(self) -> List[Resource]:
        if self.year == "2014":
            if self.split in ("train", "val"):
                if self.split == "train":
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

        return [images, annotations]

    @functools.lru_cache()
    def _load_annotations(self) -> Dict[str, Dict]:
        name = pathlib.Path(self.resources()[1].url).name
        datapipe: Iterable = (str(self.data_dir / name),)
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
        annotations = {
            image_name: dict(annotations=list(tuple(zip(*sorted(tuple(annotation.items()))))[1]))
            for image_name, annotation in annotations.items()
        }
        # Some images don't have annotations
        annotations.update(
            {key: dict(annotations=[]) for key in set(image_id_to_name.values()) - set(annotations.keys())}
        )
        return annotations

    @property
    def dp(self) -> Iterable[Dict[str, Any]]:
        name = pathlib.Path(self.resources()[0].url).name
        datapipe: Iterable = (str(self.data_dir / name),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromZip(datapipe)

        return dp.iter.Map(
            datapipe,
            functools.partial(self._default_collate_sample, annotations=self._load_annotations()),
        )
