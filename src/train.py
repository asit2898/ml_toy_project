import os
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import wandb
from datasets import TrainDataSet, ValDataSet, TestDataSet
import json

from models.simple_cnn import SimpleCNN
from utils.utils import circle_prediction_accuracy
from utils.parser import CircleDetectorParser


class CircleDetector(pl.LightningModule):
    def __init__(self, iou_threshold=0.5, lr=1e-3):
        super(CircleDetector, self).__init__()
        self.model = SimpleCNN()
        self.iou_threshold = iou_threshold
        self.lr = lr
        self.val_losses = []
        self.val_accs = []
        self.test_accs = []

    def training_step(self, batch, batch_idx):
        """
        Using a MSE loss function between (center_x, center_y, radius) between predicted and ground truth
        Although final metric is IOU based accuracy, this loss function is used to train the model
        """
        x, y = batch
        x = x.unsqueeze(1)
        y_predicted = self.model(x)
        loss = F.mse_loss(y_predicted, y)

        self.log("train/loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        # Calculate loss
        x, y = batch
        x = x.unsqueeze(1)
        y_predicted = self.model(x)
        val_loss = F.mse_loss(y_predicted, y).item()

        self.log("val/loss", val_loss)

        # Calculate accuracy based on IOU
        accuracy = circle_prediction_accuracy(
            threshold=self.iou_threshold, circlesA=y_predicted, circlesB=y
        )

        self.val_losses.append(val_loss)
        self.val_accs.append(accuracy)

        return {"val_loss": val_loss, "val_acc": accuracy}

    def on_validation_epoch_end(self):
        # Aggregate metrics from all validation batches
        avg_loss = np.array(self.val_losses).mean().item()
        avg_acc = np.array(self.val_accs).mean().item()

        self.val_losses.clear()
        self.val_accs.clear()

        self.log("val/loss", avg_loss)
        self.log("val/acc", avg_acc)

    def test_step(self, batch, batch_idx):
        # Calculate test accuracy based on IOU
        x, y = batch
        x = x.unsqueeze(1)

        y_hat = self.model(x)

        accuracy = circle_prediction_accuracy(
            threshold=self.iou_threshold, circlesA=y_hat, circlesB=y
        )

        self.test_accs.append(accuracy)

        return {"test_acc": accuracy}

    def on_test_epoch_end(self):
        # Aggregate test metrics

        avg_accuracy = np.array(self.test_accs).mean().item()
        self.test_accs.clear()

        self.log("test/acc", avg_accuracy)


if __name__ == "__main__":

    # Parser with following argument priority
    # 1st: Command Line Arguments || 2nd: Config file arguments || 3rd: Argparse defaults
    # This helps keeping track of arguments through config files and allows fast command line experimentation through bash scripts

    parser = CircleDetectorParser()
    parser.add_args()

    args = parser.parse_args()
    print(f"Arguments used: \n", vars(args))

    if args.save_model_path:
        if not os.path.exists(args.save_model_path):
            os.makedirs(args.save_model_path)

    if args.output_path:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

    wandb.init(config=vars(args), project=args.wandb_project_name)
    args = wandb.config
    logger = WandbLogger(log_model=True)

    # Set seed
    pl.seed_everything(args.seed)

    # Create dataloaders
    train_dataloader = DataLoader(
        TrainDataSet(
            epoch_length=args.train_epoch_length, noise_level=args.noise_level
        ),
        batch_size=args.train_batch_size,
    )
    val_dataloader = DataLoader(ValDataSet(), batch_size=args.val_batch_size)
    test_dataloader = DataLoader(TestDataSet(), batch_size=args.val_batch_size)

    # Create callbacks

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.save_model_path,
        monitor="val/acc",
        mode="max",
        save_top_k=args.save_top_k,
        verbose=True,
        save_last=True,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val/acc", min_delta=0.00, patience=3, verbose=False, mode="max"
    )

    # Create trainer

    model = CircleDetector(iou_threshold=args.iou_threshold, lr=args.lr)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=args.max_epochs,
        val_check_interval=args.val_check_interval,
        fast_dev_run=args.fast_dev_run,
        precision=64,
    )

    # Train model
    if args.mode == "train":
        trainer.fit(
            model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )

    # Get test metrics on best model
    if args.mode == "train":
        best_model_path = checkpoint_callback.best_model_path

    if args.mode == "test":
        if args.model_path == None:
            raise ValueError("model_path not specified")
        else:
            best_model_path = args.model_path

    trained_model = CircleDetector.load_from_checkpoint(best_model_path)

    results = trainer.test(model=trained_model, dataloaders=test_dataloader)

    print(f"Final test results:\n {results}")
