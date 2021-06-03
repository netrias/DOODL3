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

## Training and testing

Model training was carried out in `model-runs.ipynb`. Inside this notebook, MNIST and Iris train/test splits were loaded using functions from `data_loaders.py`. The notebook outputs model weights and predictions to subdirectories under a "data" directory. Note that the model weights and predictions that we used to generate our figures are included as a zip file, so there is no need to rerun `model_runs.ipynb` unless you want to conduct the training and testing again.

## Creating plots for test predictions and weights

The `plotting-results.ipynb` notebook generates violinplots of model accuracy, prediction entropy, and model weight distributions. 

The following figures were generated using this notebook: Figure 2, Figure 4, Figure S.4, Figure S.5, Figure S.9.

## Generating Morse graphs and order retraction plots

The notebooks `morse_graphs_and_order_retraction_iris.ipynb` and `morse_graphs_and_order_retraction_mnist.ipynb` can be used to generate morse graphs and order retraction plots for different model architectures. These notebooks pull model run data from the "data" directory. Note that you must restart the notebook kernel each time you want to generate a new morse graph and order retraction plot (see notebook instructions). Otherwise data from previous morse graphs and order retraction computations will leak into the new computations. Also note that the color mapping towards the ends of these notebooks must be done manually.

The following figures were generated using these notebooks: Figure 3, Figure 5 (left-most plot), Figure S.1

## Deep dives on alignment of model weights with order retraction plots:

The notebook `order_retraction_deep_dive_mnist.ipynb` was used to inspect metrics of model weights that appeared in different regions of the order retraction plot associated with the MNIST model architecture that uses 8 epochs.

The following figures were generated using this notebook: Figure 5 (second through fourth plots), Figure S.2

## Analysis of CMGDB run time with different parameters

The notebooks `timing_analysis_num_weights.ipynb` and `timing_analysis_phase_subdiv.ipynb` were used to measure run times with different numbers of weights and phase subdivisions. 

The following figures were generated using these notebooks: Figure S.6, Figure S.7

## Contributing

MIT License.
