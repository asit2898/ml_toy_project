import torch
from torch.utils.data import Dataset
import numpy as np

from data.data_utils import noisy_circle
from typing import Optional


class TrainDataSet(Dataset):
    """
    Uses noisy_circle function from data_utils.py to generate training data of user specified epoch length
    """

    def __init__(
        self,
        epoch_length: int = 50000,
        noise_level: float = 0.5,
        img_size: int = 128,
        min_radius: Optional[float] = None,
        max_radius: Optional[float] = None,
    ):

        assert img_size == 128, "Currently code only supports images of size 128*128"

        self.epoch_length = epoch_length
        self.noise_level = noise_level
        self.img_size = img_size

        if not min_radius:
            min_radius = img_size // 10
        if not max_radius:
            max_radius = img_size // 2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        img, circle = noisy_circle(
            img_size=self.img_size,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            noise_level=self.noise_level,
        )
        return (
            torch.tensor(img, dtype=torch.float64),
            torch.tensor([circle.row, circle.col, circle.radius], dtype=torch.float64),
        )


class ValDataSet(Dataset):
    """
    Generates validation data from the validation_data_images.npy and validation_data_circles.npy files
    """

    def __init__(self):
        self.images = np.load("datasets/validation_data_images.npy")
        self.circles = np.load("datasets/validation_data_circles.npy")

        assert len(self.images) == len(
            self.circles
        ), "Please check the validation data. Number of images and circles do not match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx], dtype=torch.float64),
            torch.tensor(self.circles[idx], dtype=torch.float64),
        )


class TestDataSet(Dataset):
    """
    Generates test data from the test_data_images.npy and test_data_circles.npy files
    """

    def __init__(self):
        self.images = np.load("datasets/test_data_images.npy")
        self.circles = np.load("datasets/test_data_circles.npy")

        assert len(self.images) == len(
            self.circles
        ), "Please check the test data. Number of images and circles do not match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx], dtype=torch.float64),
            torch.tensor(self.circles[idx], dtype=torch.float64),
        )
