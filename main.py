import more_itertools
import tqdm

from torchvision import datasets

from utils import use_dataset_mocks

datasets.home("~/datasets")

with use_dataset_mocks():
    dataset = datasets.load("caltech101")
    datasets.show_example(dataset)
    more_itertools.consume(tqdm.tqdm(dataset))
