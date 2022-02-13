# qr_for_landcover


This repository details and replicates the experimental steps for the landcover experiments for ``Resolving Label Uncertaintiy with Implicit Generative Models."

The repository is still in progress (as the EnviroAtlas experiment protocol gets incorporated into the torchgeo package), as noted by the TODO's below. Currently this repo supports the experiments for the Chesapeake datasets, and EnviroAtlas experiments will be supported soon. Thank you for your patience.

### Setting up the environment
 1. From the qr_for_landcover directory run `conda env create -f environment.yml`
 2. Activate the environment by running `conda activate qr_torchgeo`
 3. Update to the most recent torchgeo by running `python -m pip install git+https://github.com/microsoft/torchgeo`.

### Downloading datasets
The default parameters in this repo will assume you have data stored in ~/torchgeo_data. To do this, you can follow these steps:
Chesapeake:
- First download the ChesapeakeCVPR dataset, e.g. by following the instructions at: https://lila.science/faq#downloadtips . In what follows we assume the data is housed in `/torchgeo_data/` .
- Next add the prior files by downloading and following the instrcutions from downloading https://zenodo.org/record/5866525#.YeXhCVjMIws (use version 1.1).
  
EnviroAtlas:
- The files for the EnviroAtlas dataset used in this analysis can be downloaded at: https://zenodo.org/record/5778193#.YeYSCVjMIws. 


### Constructing the priors (optional):
Important: If you want to skip constructing the priors and move ahead to the experiment scripts, you can download the precomputed priors from torchgeo using the steps in the previous section. You only need to follow these steps if you want to recreate the priors from the original data.



To construct the priors for the Chesapeake dataset, first make sure you have the original dataset downloaded via torchgeo. Then, from `qr_for_landcover/compute_priors' run:
1. `compute_cooccurrence_matrices_chesapeake.py` to compute the class cooccurrence matrices from the training sets in each state, and then 
2. `make_priors_chesapeake.py` to make the priors and save them in the folder from torchgeo.
Note that you'll need to change the paths to the data directories at the top of each script. The notebooks in the `qr_for_landcover/compute_priors` will visualize these outputs. 

To construct the priors for the EnviroAtlas dataset, theres a few additional steps to download the additional from the original data sources. The quick way is to download the data in the zip file from torchgeo (see above). 

0. TODO: explain gathering the raw data that gets input to the priors.
1. The cooccurrence matrices for the EnviroAtlas data are provided in this repo, so you don't have to download the full data to use them. If you're interested, the `compute_cooccurrence_matrices_envirotlas.py` is the script to generate them from the full EnviroAtlas data. 
2. `make_priors_envirotlas.py` makes the priors and saves them in the folder from torchgeo.
Note that you'll need to change the paths to the data directories at the top of each script. The notebooks in the `qr_for_landcover/compute_priors` will visualize these outputs. 

To learn the Enviroatlas prior from its inputs: 
1. First run `earn_the_prior_enviroatlas.py` from the `experiment_scripts` folder. 
2. To run the model forward and save these learned priors, from the `evaluation` folder run `save_learned_priors.py`
3. You can visualize the learned priors with `evaluation/visualize_output/visualize_learned_priors_ea.ipynb`.

### Experiment Scripts:
The experiment scripts are broken up into hyperparameter search scripts (hp_\*.py) and evaluation runs (run_\*.py). 

Chesapeake:
- To run the Chesapeake full experiment with hyperparameter search, run `hp_gridsearch_de.py`, then `run_qr_in_chesapeake_north.py.` 

EnviroAtlas (in progress - integrating with torchgeo, currently not functional):
- To train the EnviroAtlas high-res model with just NAIP Imagery as input and with the prior concatenated as input, run `hp_gridsearch_pittsburgh.py` and `hp_gridsearch_pittsburgh_with_prior_as_input.py` (then evaluate the best model using the evaluation scripts below)
- To train the EnviroAtlas QR model with the best pittburgh model as the initializing for the model weights, run `hp_gridsearch_qr_from_checkpoint_pittsburgh.py` to pick parameters in Pittsburgh and `run_qr_forward_enviroatlas_from_checkpoint.py` to run the model in the test set in each city. 
- To train the EnviroAtlas QR model with random initialization of the model weights, run `hp_gridsearch_qr_from_scratch_pittsburgh.py` to pick parameters in Pittsburgh and `run_qr_forward_enviroatlas_from_sctrach.py` to run the model in the test set in each city. 
- To use train the EnviroAtlas model using the learned prior, run `hp_gridsearch_qr_learned_prior_from_checkpoint_pittsburgh.py` and `run_qr_forward_enviroatlas_learned_prior_from_checkpoint.py`.

### Evaluating and Visualizing results:
To evaluate the Chesapeake Conservancy predictions in NY and PA:
1. Run `save_predictions_chesapeake.py` or `save_predictions_enviroatlas.py` from the `evaluation` folder. If you only want to evaluate some enviroatlas experiments, you'll have do comment out some lines in that script.
2. Evaluate the predictions against the high resolution labels with `evaluation/evaluate_qr_models_chesapeake.ipynb`

To evaluate the EnviroAtlas predictions in each state:
1. Run  `save_predictions_envirotlas.py` from the `evaluation` folder.
2. Evaluate the predictions against the high resolution labels with `evaluation/evaluate_enviroatlas_hr_models.ipynb` (for the models trained with high resolution label data) and `evaluation/evaluate_qr_models_enviroatlas.ipynb` (for the QR and RQ models).

To visualize the outputs, use the notebooks in `evaluation/visualization.`

Notebooks to generate the figures in the paper are in the `figure_notebooks` folder.


