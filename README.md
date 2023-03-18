
<h1 align="center">ML toy project: Circle Detector</h1>

***
<p align="center">
 A simple VGGnet like CNN based circle detector. This is a toy example to consolidate good coding practices and create a starting template for more complicated projects using PyTorch lightning and wandb.
</p>

***
<h2 style="font-size: 24px; font-weight: 400;" align="center"> Guide To Run</h2>


## Installation
Clone the repo and cd into the folder. It is recommended to start with a fresh conda environment to avoid package conflicts. Then install the requirements:
```
pip install -r requirements.txt
```
Install a package in editable mode from the current directory"
```
pip install -e .
```

## Generating Validation and Test Data
You can either download the datasets folder from [here](https://drive.google.com/drive/folders/1NkaQTNSLVR-8JGmZ8UwOFKh6RWcxhPQ3?usp=sharing) to reproduce the results or you can generate your own test and validation datasets using:

```
python3 data/generate_data.py
```

## Evaluating the best model 
To get the test accuracy using the best model run the following command:
```
python3 src/train.py --accelerator cpu --devices 1 --mode test --model_path best_model.ckpt
```

## Training a new model

| Priority | Source |
| --- | --- |
| top | User-specified command line arguments |
| middle | Arguments from a config file provided using the `--config_file` flag |
| least | Argparse default arguments from `utils/parser.py` |

It is recommended to have a set of configurations in a config file, and then change one or two parameters quickly through the command line for fast experimentation. 

To train the model using the configs used for the best_model use the following command:
```
python3 src/train.py --config_file configs/best_train.json
```

## Conducting hyperparameter sweeps
Modify the sweep config file at `src/sweep.yaml`. Then run:
```
wandb sweep src/sweep.yaml
```
Then start the wandb agent above commmand's output. You can run multiple runs for the agent using *--count* flag 

<h2 style="font-size: 24px; font-weight: 400;" align="center"> Model Architecture & Training</h2>

This is a VGG net like model with reduced number of parameters and built for a single channel. The model takes input of shape *(batch_size,128, 128)* and outputs of shape *(batch_size,3)* where each output corresponds to the predicted *(center x, center y, radius)* of the circle in the image.

The model is trained using Mean Squared Error loss. A sweep was conducted over lr and batch_size (config can be found in `src/sweep.yaml`). Validation accuracy was used to select the best model.

The model is trained for 10 max_epochs and undergoes early stopping based on validation accuracy. IOU threshold is set to 0.5. 


<h2 style="font-size: 24px; font-weight: 400;" align="center"> Results</h2>

For the best model:
<p align="center">

| Metric | Value|
|---|---|
| Train Loss | 40.417 |
| Validation Loss | 49.806 |
| Validation Accuracy | 99.115 |
| Test Accuracy | 99.062 |
</p>


*Do note that the train, val and test sets are from the exact same image generating probability distribution so the high accuracy numbers are to be expected*

