# qr_for_landcover


This repository details and replicates the experimental steps for the landcover experiments for ``Resolving Label Uncertaintiy with Implicit Generative Models."

The repository is still in progress, as noted by the TODO's below. Thank you for your patience.

TODO - link to paper once available

### Setting up the environment
 1. From the qr_for_landcover directory run `conda env create -f environment.yml`
 2. Activate the environment by running `conda activate qr_for_landcover`

### Downloading datasets
TODO - how to download the datasets from torchgeo.

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

To learn the Enviroatlas prior from its inputs: 
1. First run 'learn_the_prior_enviroatlas.py` from the `experiment_scripts` folder. 
2. To run the model forward and save these learned priors, from the `evaluation` folder run `save_learned_priors.py`
3. You can visualize the learned priors with `evaluation/visualize_output/visualize_learned_priors_ea.ipynb`.

### Experiment Scripts:
The experiment scripts are broken up into hyperparameter search scripts (hp_\*.py) and evaluation runs (run_\*.py). 
- To run the Chesapake full experiment with hyperparameter search, run `hp_gridsearch_de.py`, then `run_qr_in_chesapeake_north.py.` 
- To train the EnviroAtlas high-res model with just NAIP Imagery as input and with the prior concatenated as input, run `hp_gridsearch_pittsburgh.py` and `hp_gridsearch_pittsburgh_with_prior_as_input.py` (then evaluate the best model using the evaluation scripts below)
- To train the EnviroAtlas QR model with the best pittburgh model as the initializing for the model weights, run `hp_gridsearch_qr_from_checkpoint_pittsburgh.py` to pick parameters in Pittsburgh and `run_qr_forward_enviroatlas_from_checkpoint.py` to run the model in the test set in each city. 
- To train the EnviroAtlas QR model with random initialization of the model weights, run `hp_gridsearch_qr_from_scratch_pittsburgh.py` to pick parameters in Pittsburgh and `run_qr_forward_enviroatlas_from_sctrach.py` to run the model in the test set in each city. 
- To use train the EnviroAtlas model using the learned prior, run `hp_gridsearch_qr_learned_prior_from_checkpoint_pittsburgh.py` and `run_qr_forward_enviroatlas_learned_prior_from_checkpoint.py`.

### Evaluating and Visualizing results:
To evaluate the Chesapeake Conservancy predictions in NY and PA:
1. Run `save_predictions_chesapeake.py` from the `evaluation` folder.
2. Evaluate the predictions against the high resolution labels with `evaluation/evaluate_qr_models_chesapeake.ipynb`

To evaluate the EnviroAtlas predictions in each state:
1. Run  `save_predictions_envirotlas.py` from the `evaluation` folder.
2. Evaluate the predictions against the high resolution labels with `evaluation/evaluate_enviroatlas_hr_models.ipynb` (for the models trained with high resolution label data) and `evaluation/evaluate_qr_models_enviroatlas.ipynb` (for the QR and RQ models).

To visualize the outputs, use the notebooks in `evaluation/visualization.`

Notebooks to generate the figures in the paper are in the `figure_notebooks` folder.


