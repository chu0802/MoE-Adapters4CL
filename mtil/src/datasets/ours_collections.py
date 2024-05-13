from .collections import ClassificationDataset
import json


class OursClassificationDataset(ClassificationDataset):
    def split_dataset(self):
        with (self.root / self.annotation_filename).open("r") as f:
            data = json.load(f)