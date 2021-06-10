import more_itertools
import tqdm

from torchvision import datasets
from utils import use_dataset_mocks

datasets.home("~/datasets")

with use_dataset_mocks():
    dataset = datasets.get("caltech256")
    dataset.show_example()
    more_itertools.consume(tqdm.tqdm(dataset.as_datapipe()))
