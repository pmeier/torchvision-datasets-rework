import csv
import pathlib
from typing import Optional

__all__ = ["Meta"]


class Meta:
    def __init__(
        self,
        *,
        citation: Optional[str] = None,
        homepage: Optional[str] = None,
        license: Optional[str] = None,
        num_samples: Optional[int] = None,
        classes: Optional[pathlib.Path] = None,
    ) -> None:
        if isinstance(classes, pathlib.Path):
            with open(classes, newline="") as fh:
                classes = {int(row["label"]): row["class"] for row in csv.DictReader(fh)}
        self.classes = classes
        if classes:
            self.labels = {cls: label for label, cls in classes.items()}
