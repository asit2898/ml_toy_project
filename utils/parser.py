from argparse import ArgumentParser
import sys as _sys
import os
import json


class CircleDetectorParser(ArgumentParser):
    """
    Class that lists all the arguments to be parsed for the circle detection model.
    Separate class helps list args into different arguments groups (ex- https://github.com/facebookresearch/BLINK/blob/main/blink/common/params.py)
    Also overwrite the argparse defauts with the config file values
    Final priority of arguments:
    1st: Command Line Arguments || 2nd: Config file arguments || 3rd: Argparse defaults
    """

    def __init__(self, description="Arguments for circle detection"):
        super().__init__(description=description)

    def parse_args(self) -> dict:
        """
        First looks if a config file was provided and overrides the argparse default values
        Then parses the arguments and returns them as a dictionary

        This helps maintain the following priority of arguments:
        1st: Command Line Arguments || 2nd: Config file arguments || 3rd: Argparse defaults
        """
        user_args = _sys.argv

        try:
            i = user_args.index("--config_file")
        except ValueError:
            i = None

        if i and i + 1 < len(user_args):
            f = user_args[i + 1]
            if os.path.exists(f):
                with open(f, "r") as f:
                    config_args = json.load(f)
                self.set_defaults(**config_args)

        args = super().parse_args()
        return args

    def add_args(self):
        # Add all arguments
        self.add_train_args()
        self.add_other_args()

    def add_train_args(self):
        """
        Adds all training arguements to the parser
        """
        parser = self.add_argument_group("Arguments")

        parser.add_argument(
            "--fast_dev_run",
            type=bool,
            default=False,
            help="Runs pytroch lightning's fast_dev_run for debugging using single batch",
        )

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
            help="Path to save the trained model",
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
            "--train_epoch_length",
            type=int,
            default=50000,
            help="Length of training dataset generated",
        )

        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

        parser.add_argument(
            "--patience", type=int, default=1, help="Patience for early stopping"
        )

    def add_other_args(self):
        """
        Adds all other arguments to the parser
        """
        parser = self.add_argument_group("Other Arguments")

        # Do not change the default value of config_file to anything other than None
        # If you want to use a config file, pass the path to the config file using the --config_file argument
        # Otherwise the config file will be ignored
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
            "--model_path",
            type=str,
            default=None,
            help="Path to the saved model. If mode is test, this arguments is needed.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=1048,
            help="Random seed to set all seeds using pytorch_lightning",
        )

        parser.add_argument(
            "--iou_threshold",
            type=float,
            default=0.5,
            help="IOU threshold for accuracy evaluation",
        )

        parser.add_argument(
            "--output_path",
            type=str,
            default="outputs",
            help="Path to save all outputs",
        )

        parser.add_argument(
            "--noise_level",
            type=float,
            default=0.5,
            help="Noise level used to create the training set",
        )

        parser.add_argument(
            "--wandb_project_name",
            type=str,
            default=None,
            help="Name of the wandb project to log to",
        )

        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of workers to use for dataloading",
        )
