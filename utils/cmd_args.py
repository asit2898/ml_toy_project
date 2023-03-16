from argparse import ArgumentParser


class CircleDetectorParser(ArgumentParser):
    """
    Class that lists all the arguments to be parsed for the circle detection model.
    Separate class helps list args into different arguments groups for easy readability
    example from:https://github.com/facebookresearch/BLINK/blob/main/blink/common/params.py
    """

    def __init__(self, description="Arguments for circle detection"):
        super(self, CircleDetectorParser).__init__(description=description)

        # Add all arguments
        self.add_train_args()
        self.add_other_args()

    def add_train_args(self):
        """
        Adds all training arguements to the parser
        """
        parser = self.add_argument_group("Arguments")

        parser.add_argument(
            "--train_batch_size", type=int, default=32, help="Batch size for training"
        )
        parser.add_argument(
            "--val_batch_size", type=int, default=32, help="Batch size for validation"
        )
        parser.add_argument(
            "--save_model_path",
            type=str,
            default="outputs/models",
            help="Path to save the trained model and other outputs",
        )
        parser.add_argument(
            "--save_top_k", type=int, default=1, help="Number of best models to save"
        )
        parser.add_argument(
            "--accelerator",
            type=str,
            default="cpu",
            help="Type of accelerator to use, options:[cpu, gpu]",
        )
        parser.add_argument(
            "--devices",
            type=int,
            default=1,
            help="Number of devices to use (only makes sense for GPU)",
        )
        parser.add_argument(
            "--max_epochs",
            type=int,
            default=100,
            help="Maximum number of epochs to train for",
        )
        parser.add_argument(
            "--val_check_interval",
            type=float,
            default=0.25,
            help="Fraction of training epoch to run validation",
        )
        parser.add_argument(
            "--use_wandb",
            type=bool,
            default=False,
            help="Whether to use wandb. Defaults to tensorboard otherwise.",
        )
        parser.add_argument(
            "--train_epoch_length",
            type=int,
            default=50000,
            help="Length of training dataset generated",
        )

        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    def add_other_args(self):
        """
        Adds all other arguments to the parser
        """
        parser = self.add_argument_group("Other Arguments")

        parser.add_argument(
            "--config_file",
            type=str,
            default=None,
            help="json file where arguments are provided. These WILL be OVERWRITTEN by command line arguments if provided.",
        )

        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            help="Mode of the script. options: [train, test]",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=1048,
            help="Random seed to set all seeds using pytorch_lightning",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=None,
            help="Path to the saved model. If mode is test, this arguments is needed.",
        )
        parser.add_argument(
            "--iou_threshold",
            type=float,
            default=0.5,
            help="IOU threshold for accuracy evaluation",
        )
