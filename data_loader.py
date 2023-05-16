import os
import random
from glob import glob
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TrainDataset(Dataset):

    def __init__(self, data: List[str], target_size=(64, 64)):
        """
        Loads normal IXI images from data_dir

        @param data:
            paths to images
        @param: target_size: tuple (int, int), default: (64, 64)
            the desired output size
        """
        super(TrainDataset, self).__init__()
        self.target_size = target_size
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.data[idx]).convert('L')
        # Pad to square
        img = transforms.Pad(((img.height - img.width) // 2, 0), fill=0)(img)
        # Resize
        img = img.resize(self.target_size, Image.BICUBIC)
        # Convert to tensor
        img = transforms.ToTensor()(img)

        return img


class TrainDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, target_size=(64, 64), batch_size: int = 32, train_val_split: float = 0.8):
        """
        Data module for training

        @param data_dir: str
            path to directory containing data
        @param: target_size: tuple (int, int), default: (64, 64)
            the desired output size
        @param: batch_size: int, default: 32
            batch size
        """
        super(TrainDataModule, self).__init__()
        self.target_size = target_size
        self.batch_size = batch_size

        # Load files from data_dir
        data = glob(data_dir + '/*.png')
        assert len(data) > 0, f'No files found in {data_dir}'
        print(f'Found {len(data)} files in {data_dir}')

        # Shuffle data
        original_seed = random.getstate()
        random.seed(42)
        random.shuffle(data)
        random.setstate(original_seed)

        # Split data
        split_idx = int(len(data) * train_val_split)
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]

    def train_dataloader(self):
        return DataLoader(TrainDataset(self.train_data, self.target_size),
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(TrainDataset(self.val_data, self.target_size),
                          batch_size=self.batch_size,
                          shuffle=False)


class TestDataset(Dataset):

    def __init__(self, img_csv: str, pos_mask_csv: str, neg_mask_csv: str, target_size=(64, 64)):
        """
        Loads anomalous images, their positive masks and negative masks from data_dir

        @param img_csv: str
            path to csv file containing filenames to the images
        @param img_csv: str
            path to csv file containing filenames to the positive masks
        @param img_csv: str
            path to csv file containing filenames to the negative masks
        @param: target_size: tuple (int, int), default: (64, 64)
            the desired output size
        """
        super(TestDataset, self).__init__()
        self.target_size = target_size
        self.img_paths = pd.read_csv(img_csv)['filename'].tolist()
        self.pos_mask_paths = pd.read_csv(pos_mask_csv)['filename'].tolist()
        self.neg_mask_paths = pd.read_csv(neg_mask_csv)['filename'].tolist()

        assert len(self.img_paths) == len(self.pos_mask_paths) == len(self.neg_mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.img_paths[idx]).convert('L')
        img = img.resize(self.target_size, Image.BICUBIC)
        img = transforms.ToTensor()(img)

        # Load positive mask
        pos_mask = Image.open(self.pos_mask_paths[idx]).convert('L')
        pos_mask = pos_mask.resize(self.target_size, Image.NEAREST)
        pos_mask = transforms.ToTensor()(pos_mask)

        # Load negative mask
        neg_mask = Image.open(self.neg_mask_paths[idx]).convert('L')
        neg_mask = neg_mask.resize(self.target_size, Image.NEAREST)
        neg_mask = transforms.ToTensor()(neg_mask)

        return img, pos_mask, neg_mask


def get_test_dataloader(split_dir: str, pathology: str, target_size: Tuple[int, int], batch_size: int):
    """
    Loads test data from split_dir

    @param split_dir: str
        path to directory containing the split files
    @param pathology: str
        pathology to load
    @param batch_size: int
        batch size
    """
    img_csv = os.path.join(split_dir, f'{pathology}.csv')
    pos_mask_csv = os.path.join(split_dir, f'{pathology}_ann.csv')
    neg_mask_csv = os.path.join(split_dir, f'{pathology}_neg.csv')

    return DataLoader(TestDataset(img_csv, pos_mask_csv, neg_mask_csv, target_size),
                      batch_size=batch_size,
                      shuffle=False,
                      drop_last=False)


def get_all_test_dataloaders(split_dir: str, target_size: Tuple[int, int], batch_size: int):
    """
    Loads all test data from split_dir

    @param split_dir: str
        path to directory containing the split files
    @param batch_size: int
        batch size
    """
    pathologies = [
        'absent_septum',
        'artefacts',
        'craniatomy',
        'dural',
        'ea_mass',
        'edema',
        'encephalomalacia',
        'enlarged_ventricles',
        'intraventricular',
        'lesions',
        'mass',
        'posttreatment',
        'resection',
        'sinus',
        'wml',
        'other'
    ]
    return {pathology: get_test_dataloader(split_dir, pathology, target_size, batch_size)
            for pathology in pathologies}
