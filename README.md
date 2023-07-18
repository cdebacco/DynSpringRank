# Dynamic SpringRank
Python implementation of Dynamic SpringRank  model described in:
-  *A model for efficient dynamical ranking in networks*  
    Andrea Della Vecchia, Kibidi Neocosmos, Daniel B. Larremore, Cristopher Moore, and Caterina De Bacco

Dynamic SpringRank is a physics-inspired method for inferring dynamic rankings in directed temporal networks —
networks in which each directed and timestamped edge reflects the outcome and timing of a pairwise
interaction. It is the natural extension of [SpringRank](https://arxiv.org/abs/1709.09002) to temporal networks. 

<!-- If you use this code, please cite [1] -->

## What's included?
- `src` : contains the Python implementation of two versions of Dynamic SpringRank (Online and Offline) as well as SpringRank.
- `data/input` : contains synthetic data used to illustrate the functioning of the models. Note: 'static' data in folder refers to data without a meaningful time component (refer to 'The Relevance of Time' section in paper for further details)
- `data/output` : location for saved results after running models.

## Requirements
The project was developed in Python 3.8.11 with the packages contained in the *requirements.txt*. We recommend creating a conda environment and installing the pre-requisite packages with `conda create --name DSR --file requirements.txt`

## Usage
The model can be run by executing the `run_models.py` while in the `src` directory. The `run_models.py` script has the following arguments:

- `--model` select the model to run
- `--dataset` file name of the synthetic dataset that will be used (excluding file extension)
- `--save` flag that saves the output of the models
- `--verbose` flag that prints more details of internal procedures of model as it is running

## Input format
The model accepts as input a numpy array of shape `[T, N, N]`, where *T* is the number of timesteps and *N* is the number of nodes in the network.

## Output
The model outputs a Python Tuple containing the evaluations metrics and rankings of the nodes. Note: when the `--save` flag is called, the output is saved as a Python Dictionary containing the aforementioned results as well as the runtime of the model. 