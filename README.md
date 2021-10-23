# qr_for_landcover

### Constructing the priors:
If you want to skip constructing the priors and move ahead to the experiment scripts, you can download the precomputed priors from torchgeo (todo: add link).

To construct the priors for the Chesapeake dataaset, first make sure you have the original dataset downloaded via torchgeo. Then, from `qr_for_landcover/compute_priors' run:
1. `compute_cooccurrence_matrices_chesapeake.py` to compute the class cooccurrence matrices from the training sets in each state, and then. 
2. `make_priors_chesapeake.py` to make the priors and save them in the folder from torchgeo.
Note that you'll need to change the paths to the data directories at the top of each script. The notebooks in the `qr_for_landcover/compute_priors` will visualize these outputs. 

### Experiment Scripts:
The experiment scripts are breaken up into hyperparameter search scripts (hp_\*.py) and evaluation runs (run_\*.py). 


hp_gridsearch_de.py                              hp_gridsearch_qr_from_scratch_pittsburgh.py    run_qr_in_chesapeake_north.py
hp_gridsearch_pittsburgh.py                      learn_the_prior_enviroatlas.py                 train.py
hp_gridsearch_pittsburgh_with_prior_as_input.py  run_qr_forward_enviroatlas_from_checkpoint.py
hp_gridsearch_qr_from_checkpoint_pittsburgh.py   run_qr_forward_enviroatlas_from_scratch.py

