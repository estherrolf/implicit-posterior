# qr_for_landcover


This repository details and replicates the experimental steps for the landcover experiments for ``Resolving Label Uncertaintiy with Implicit Generative Models."

The repository is still in progress, as noted by the TODO's below. Thank you for your patience.

TODO - link to paper once available

TODO - explain how to make the python environment.

### Constructing the priors:
If you want to skip constructing the priors and move ahead to the experiment scripts, you can download the precomputed priors from torchgeo (todo: add link).

To construct the priors for the Chesapeake dataset, first make sure you have the original dataset downloaded via torchgeo. Then, from `qr_for_landcover/compute_priors' run:
1. `compute_cooccurrence_matrices_chesapeake.py` to compute the class cooccurrence matrices from the training sets in each state, and then 
2. `make_priors_chesapeake.py` to make the priors and save them in the folder from torchgeo.
Note that you'll need to change the paths to the data directories at the top of each script. The notebooks in the `qr_for_landcover/compute_priors` will visualize these outputs. 

To construct the priors for the EnviroAtlas dataset, theres a few additional steps to download the additional from the original data sources. The quick way is to download the data in the zip file from torchgeo (TODO: add instructions on this). 
1. The cooccurrence matrices for the EnviroAtlas data are provided in this repo, so you don't have to download the full data to use them. If you're interested, the `compute_cooccurrence_matrices_envirotlas.py` is the script to generate them from the full EnviroAtlas data. 
2. `make_priors_envirotlas.py` makes the priors and saves them in the folder from torchgeo.
Note that you'll need to change the paths to the data directories at the top of each script. The notebooks in the `qr_for_landcover/compute_priors` will visualize these outputs. 

To learn the Enviroatlas prior from its inputs, 
[TODO -- reference script to train the learned priors and script to generate and save priors from the models.]

### Experiment Scripts:
The experiment scripts are broken up into hyperparameter search scripts (hp_\*.py) and evaluation runs (run_\*.py). 
- To run the Chesapake full experiment with hyperparameter search, run `hp_gridsearch_de.py`, then `run_qr_in_chesapeake_north.py.` 
- To train the EnviroAtlas high-res model with just NAIP Imagery as input and with the prior concatenated as input, run `hp_gridsearch_pittsburgh.py` and `hp_gridsearch_pittsburgh_with_prior_as_input.py` (then evaluate the best model using the evaluation scripts below)
- To train the EnviroAtlas QR model with the best pittburgh model as the initializing for the model weights, run `hp_gridsearch_qr_from_checkpoint_pittsburgh.py` to pick parameters in Pittsburgh and `run_qr_forward_enviroatlas_from_checkpoint.py` to run the model in the test set in each city. 
- To train the EnviroAtlas QR model with random initialization of the model weights, run `hp_gridsearch_qr_from_scratch_pittsburgh.py` to pick parameters in Pittsburgh and `run_qr_forward_enviroatlas_from_sctrach.py` to run the model in the test set in each city. 
- To use train the EnviroAtlas model using the learned prior, run `hp_gridsearch_qr_learned_prior_from_checkpoint_pittsburgh.py` and `run_qr_forward_enviroatlas_learned_prior_from_checkpoint.py`.

### Evaluating and Visualizing results:
[TODO]
