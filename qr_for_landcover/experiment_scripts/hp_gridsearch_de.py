"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Runs the train script with a grid of hyperparameters.
"""
import itertools
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0]
TEST_MODE = False  # if False then print out the commands to be run, if True then run

# Hyperparameter options
training_set_options = ["de"]
model_options = ['fcn']
lr_options = [1e-4,1e-5,1e-3]

loss_options = ['qr_forward', 'qr_reverse']
prior_version_options = [
                       'from_cooccurrences_101_31_no_osm_no_buildings',
                        ]
additive_smooth_options = [1e-4] 
prior_smooth_options = [1e-4]

train_set, val_set, test_set = ['test', 'test', 'test']

def do_work(work, gpu_idx):
    while not work.empty():
        experiment = work.get()
        experiment = experiment.replace("GPU", str(gpu_idx))
        print(experiment)
        if not TEST_MODE:
            subprocess.call(experiment.split(" "))
    return True


def main():

    work = Queue()

    for (states_str, model, loss, prior_version, lr, additive_smooth, prior_smooth) in itertools.product(
        training_set_options,
        model_options,
        loss_options,
        prior_version_options,
        lr_options,
        additive_smooth_options,
        prior_smooth_options
    ):
        experiment_name = f"{states_str}_{model}_{lr}_{loss}_{prior_version}_additive_smooth_{additive_smooth}_prior_smooth_{prior_smooth}"

        
        train_splits = [f'{state}-{train_set}' for state in states_str.split('+')]
        val_splits = [f'{state}-{val_set}' for state in states_str.split('+')]
        test_splits = [f'{state}-{test_set}' for state in states_str.split('+')]
        
        output_dir = "../output_rep/hp_search_de/hp_search_de"
        command = (
            f"python train.py program.overwrite=True config_file=../conf/chesapeake_learn_on_prior.yml"
            + f" experiment.name={experiment_name}"
            + f" experiment.module.segmentation_model={model}"
            + f" experiment.module.learning_rate={lr}"
            + f" experiment.module.loss={loss}"
            + f" experiment.module.num_filters=128"
            + f" experiment.datamodule.batch_size=128"
            + f" expertiment.datamodule.patches_per_tile=400"
            + f" experiment.module.output_smooth={additive_smooth}"
            + f" experiment.datamodule.prior_smoothing_constant={prior_smooth}"
            + f" experiment.datamodule.train_splits={train_splits}"
            + f" experiment.datamodule.val_splits={val_splits}"
            + f" experiment.datamodule.test_splits={test_splits}"
            + f" program.output_dir={output_dir}"
            + f" program.log_dir=../logs/hp_gridsearch_de"
            + " trainer.gpus=[GPU]"
        )
        command = command.strip()

        work.put(command)

    processes = []
    for gpu_idx in GPUS:
        p = Process(target=do_work, args=(work, gpu_idx))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
        
    return


if __name__ == "__main__":
    main()
