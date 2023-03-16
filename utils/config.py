"""
Having a custom config class help with setting all parameters from json config files. Argparse is then just used to overwrite some arguments.
Helps with fast experimentation through bash script for loops that just change a few parameters.

style inspired from: https://github.com/iesl/anncur
"""

import json
import warnings
from typing import Dict, Any, Optional
import os


class BaseConfig:
    """
	Base class for all config classes. Implements updating config from parameter dictionaries or json files.
	Implements saving config to json files.
	"""

    def __init__(self, filename: Optional[str] = None) -> None:
        self.config_filename = filename

    def update_from_json(self, filename: str) -> None:
        """
		Updates config from json file
		:param filename: json file
		:return: None
		"""
        if filename is None:
            raise ValueError("Filename is not specified")

        with open(filename, "r") as f:
            json_dict = json.load(f)
            self.update_from_dict(json_dict)

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
		Updates config from dictionary
		:param config_dict: dictionary with config
		:return: None
		"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Config provided has extra key {key}")

    def to_json(self) -> str:
        """
		Converts config to json string
		:return: json string
		"""
        return json.dumps(
            self.filter_serializable(self.__dict__), indent=4, sort_keys=True
        )

    def to_json_file(self, filename: str) -> None:
        """
		Saves config to json file
		:param filename: json file
		:return: None
		"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                self.filter_serializable(self.__dict__), f, indent=4, sort_keys=True
            )

    def filter_serializable(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
		Filter out non serializable objects from dictionary
		:param d: dictionary
		:return: filtered dictionary
		"""
        filtered_d = {}
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool, list, tuple)):
                filtered_d[k] = v
            elif isinstance(v, dict):
                filtered_d[k] = self.filter_serializable(v)

        return filtered_d


class Config(BaseConfig):
    """
	Config class for CircleDetector. Sets default parameters for all attributes. These defaults can be updated from json files or dictionaries.
	"""

    def __init__(self, filename=None):

        super(Config, self).__init__(filename=filename)

        self.base_res_dir = "~/circle_detector/results"
        self.exp_id = ""

        self.seed = 1234

        self.max_time = "06:23:55:00"  # 7 days - 5 minutes
        self.fast_dev_run = False  # Run a few batches from train/dev for sanity check

        self.print_interval = 10
        self.eval_interval = 800.0

        # Data specific params

        self.mode = "train"
        self.debug_w_small_data = True

        # Model/Optimization specific params
        self.use_GPU = True
        self.num_gpus = 1
        self.strategy = ""
        self.type_optimization = ""
        self.learning_rate = 0.00001
        self.weight_decay = 0.01
        self.fp16 = False

        self.ckpt_path = ""
        self.model_type = ""  # Choose between bi-encoder, cross-encoder
        self.cross_enc_type = "default"
        self.bi_enc_type = (
            "separate"
        )  # Use "separate" encoder for query/input/mention and label/entity or "shared" encoder
        self.bert_model = ""  # Choose type of bert model - bert-uncased-small etc
        self.bert_args = {}  # Some arguments to pass to bert-model when initializing
        self.lowercase = True  # Use lowercase BERT tokenizer
        self.shuffle_data = True  # Shuffle data during training
        self.path_to_model = ""
        self.encoder_wrapper_config = ""

        self.num_epochs = 4
        self.warmup_proportion = 0.01
        self.train_batch_size = 1
        self.grad_acc_steps = 4
        self.max_grad_norm = 1.0
        self.loss_type = "ce"
        self.hinge_margin = 0.5
        self.reload_dataloaders_every_n_epochs = 0
        self.ckpt_metric = "loss"
        self.num_top_k_ckpts = 2

        ## Model specific parameters
        self.embed_dim = 768
        self.pooling_type = (
            ""  # Pooling on top of encoder layer to obtain input/label embedding
        )
        self.add_linear_layer = False
        self.max_input_len = 128
        self.max_label_len = 128

        # Eval specific
        self.eval_batch_size = 1

        if filename is not None:
            self.update_from_json(filename)

        @property
        def result_dir(self):
            """
			generates a result directory name based on the config parameters
			"""

            return result_dir
