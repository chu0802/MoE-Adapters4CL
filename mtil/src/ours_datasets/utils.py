import json
from pathlib import Path

from torch.utils.data import DataLoader

from src.ours_datasets import DATASET_MAPPING
from src.ours_datasets.transform import load_transform


class DataIterativeLoader:
    def __init__(self, dataloader, device="cuda"):
        self.len = len(dataloader)
        self.dataloader = dataloader
        self.iterator = None
        self.device = device

    def init(self):
        self.iterator = iter(self.dataloader)

    def __next__(self):
        data = next(self.iterator)
        if isinstance(data, list):
            data = [d.to(self.device) for d in data]
            return data
        else:
            data = data.to(self.device)
            return data

    def __iter__(self):
        return self

    def __len__(self):
        return self.len


def build_dataloader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def build_iter_dataloader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    device="cuda",
    **kwargs,
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return DataIterativeLoader(dataloader, device=device)


def get_dataloader(
    dataset_name,
    root,
    mode,
    transform,
    sample_num=-1,
    device="cuda",
    seed=1102,
    **dataloader_config,
):
    dataset_class = DATASET_MAPPING[dataset_name]

    dataset = dataset_class(
        root,
        mode=mode,
        transform=transform,
        sample_num=sample_num,
        seed=seed,
    )

    return build_iter_dataloader(dataset, **dataloader_config, device=device)


def get_dataloaders_from_config(config, device="cuda"):
    dataloaders = {}
    train_transform, eval_transform = load_transform()

    for dataloader_type, dataloader_config in config.data.split.items():
        dataloaders[dataloader_type] = get_dataloader(
            dataset_name=config.data.name,
            root=config.data.root,
            mode=dataloader_config.split_name,
            transform=train_transform if dataloader_type == "train" else eval_transform,
            sample_num=config.data.get("sample_num", -1),
            device=device,
            **dataloader_config,
        )

    return dataloaders


def load_class_name_list(config):
    dataset_class = DATASET_MAPPING[config.data.name]
    name, annotation_filename = (
        dataset_class.dataset_name,
        dataset_class.annotation_filename,
    )

    with (Path(config.data.root) / name / annotation_filename).open("r") as f:
        data = json.load(f)

    return data["class_names"]
