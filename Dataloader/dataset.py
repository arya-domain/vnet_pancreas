import h5py
from torch.utils.data import Dataset
from Dataloader.augmentation import *
from torchvision import transforms

from transformers import BertTokenizer
import pandas as pd
import numpy as np


def normalize_val(array):
    min_val = torch.min(array)
    max_val = torch.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array


class PancreasDataset(Dataset):
    """LA Dataset
    input: base_dir -> your parent level path
           split -> "sup", "unsup" and "eval", must specified
    """

    def __init__(self, base_dir, data_dir, split, num=None, config=None):
        self.data_dir = data_dir
        self._base_dir = base_dir
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.sample_list = []
        if split == "eval":
            with open(self._base_dir + "/eval.list", "r") as f:
                self.image_list = f.readlines()
        if split == "train":
            with open(self._base_dir + "/train.list", "r") as f:
                self.image_list = f.readlines()

        self.image_list = [item.strip() for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        self.aug = config.augmentation if split != "eval" else False
        self.training_transform = transforms.Compose(
            [
                Normalise(),
                RandomCrop((96, 96, 96)),
                ToTensor(),
            ]
        )
        self.testing_transform = transforms.Compose([Normalise()])

        file_path = self._base_dir + f"/prompt/{split}.csv"
        df = pd.read_csv(file_path, names=["key", "value"])
        self.mapping_dict = dict(zip(df["key"], df["value"]))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self.data_dir + "/" + image_name + "_norm.h5", "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        prompt = self.mapping_dict[image_name]

        sample = {"data": image, "label": label}

        if not self.aug:
            # tokens
            tokens = self.tokenizer(
                prompt, truncation=True, max_length=96, padding="max_length"
            )["input_ids"]

            tokens = normalize_val(torch.tensor(np.array(tokens)))
            sample = self.testing_transform(sample)
            sample["tokens"] = tokens.to(torch.float32)
            return sample["data"], sample["label"], sample["tokens"] 
        else:
            # tokens
            tokens = self.tokenizer(
                prompt, truncation=True, max_length=96, padding="max_length"
            )["input_ids"]

            tokens = normalize_val(torch.tensor(np.array(tokens)))
            sample = self.training_transform(sample)

            # merging with tokens
            sample[0]["tokens"] = tokens.to(torch.float32)
            sample[1]["tokens"] = tokens.to(torch.float32)

        return sample
