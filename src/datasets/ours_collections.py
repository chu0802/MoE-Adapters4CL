from .collections import ClassificationDataset
import src.datasets.ours_datasets.core_dataset as ours_core_dataset


def transform_dataset(dataset_class):
    class TransformDataset(ClassificationDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = "aircraft"
            self.train_dataset = dataset_class(
                self.location, mode="train", transform=self.preprocess
            )
            self.test_dataset = dataset_class(
                self.location, mode="test", transform=self.preprocess
            )
            self.build_dataloader()
            self.classnames = self.train_dataset.classnames
            self.process_labels()
            self.templates = [self.train_dataset.template]
    return TransformDataset


Aircraft = transform_dataset(ours_core_dataset.FGVCAircraft)
DTD = transform_dataset(ours_core_dataset.DTD)
EuroSAT = transform_dataset(ours_core_dataset.EuroSAT)
Flowers102 = transform_dataset(ours_core_dataset.Flowers102)
Food101 = transform_dataset(ours_core_dataset.Food101)
OxfordPets = transform_dataset(ours_core_dataset.OxfordPets)
StanfordCars = transform_dataset(ours_core_dataset.StanfordCars)
UCF101 = transform_dataset(ours_core_dataset.UCF101)
