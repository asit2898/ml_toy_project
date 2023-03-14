import os
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from models.simple_cnn import SimpleCNN
from utils.utils import iou
from data.data_utils import CircleParams


'''
TODO: Add the following paramters to the config file:
	- lr
	- batch_size

'''

class CircleDetector(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.model = SimpleCNN()

	def training_step(self, batch, batch_idx):
		'''
		Using a MSE loss function between (center_x, center_y, radius) between predicted and ground truth
		Although final metric is IOU based accuracy, this loss function is used to train the model
		'''
		x, y = batch
		y_predicted = self.model(x)
		loss = F.mse_loss(y_predicted, y)

		return loss


	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
		return optimizer



