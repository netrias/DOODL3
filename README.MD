# Extracting Global Dynamics of Loss Landscape in Deep Learning Models

This repository is the official implementation of [Extracting Global Dynamics of Loss Landscape in Deep Learning Models](https://github.com/netrias/DOODL3).
<!--- TODO: update above link --->

## Setup, Requirements, and Installation
Clone this repository into the environment of your choice (directory, conda env, virtualenv, etc). Conda envs are recommended.

Then follow the instructions [here](https://github.com/marciogameiro/CMGDB) to install version 0.1.0 of CMGDB.

Then to install remaining requirements:

```setup
pip3 install -r requirements.txt
```

## Training and Testing

Model training was carried out in `model-runs.ipynb`. Inside this notebook, MNIST and Iris train/test splits were loaded using functions from `data_loaders.py`. The notebook outputs model weights and predictions to subdirectories under a "data" directory. Note that the model weights and predictions that we used to generate our figures are included as a zip file, so there is no need to rerun `model_runs.ipynb` unless you want to conduct the training and testing again.

## Creating plots for test predictions and weights


## Results


## Contributing

MIT License.
