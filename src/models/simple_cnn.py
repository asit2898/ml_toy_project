import torch
import torch.nn as nn
from typing import Tuple


class SimpleCNN(nn.Module):
    """
    A simple CNN model for single channel images. Like VGG16, but with much lesser parameters and addition of batch normalization.
    Expecting input of shape (batch_size, 1, 128, 128)
    output shape: (batch_size, 3)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # adding batchnorm here as alternative to input normalization
        self.bn0 = nn.BatchNorm2d(1)

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=20, kernel_size=3, stride=1, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(20)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=20, out_channels=40, kernel_size=3, stride=1, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(40)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=40, out_channels=80, kernel_size=3, stride=1, padding="same"
        )
        self.bn3 = nn.BatchNorm2d(80)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(
            in_channels=80, out_channels=160, kernel_size=3, stride=1, padding="same"
        )
        self.bn4 = nn.BatchNorm2d(160)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(
            in_channels=160, out_channels=320, kernel_size=3, stride=1, padding="same"
        )
        self.bn5 = nn.BatchNorm2d(320)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(5120, 250)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(250, 3)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the model's layers' weights. Non bias weights are initialized using kaiming initialization.
        Bias weights are initialized to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Input Shape: (batch_size, 1, 128, 128)
        Output shape: (batch_size, 3)
        """
        x = self.bn0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.maxpool5(x)

        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
