# Small
from .collections import TinyImagenet
from .ours_collections import (
    DTD,
    Aircraft,
    EuroSAT,
    Flowers102,
    Food101,
    OxfordPets,
    StanfordCars,
    UCF101,
)

# Experimental datasets
dataset_list = [
    TinyImagenet,
    Aircraft,
    DTD,
    EuroSAT,
    Flowers102,
    Food101,
    OxfordPets,
    StanfordCars,
    UCF101,
]

def show_datasets():
    print("Total: ", len(dataset_list))
    print("Dataset: (train_len, test_len, num_classes)")
    for dataset in dataset_list:
        d = dataset(None)
        print(f"{d.name}: ", d.stats())
        for i in range(3):
            print(f"T[{i}]: ", d.template(d.classnames[i]))
        

from .cc import conceptual_captions
