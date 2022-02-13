"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Runs the train script with a grid of hyperparameters.
"""
import itertools
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [2,3] 
TEST_MODE = False  # if False then print out the commands to be run, if True then run

# Hyperparameter options
training_set_options = ["pittsburgh_pa-2010_1m"]
model_options = ['fcn']
lr_options = [1e-5,1e-4,1e-3,1e-2]

loss_options = ['ce']
additive_smooth_options = [1e-8]

train_set, val_set, test_set = ['train', 'val', 'val']

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

    for (states_str, model, lr,loss, additive_smooth) in itertools.product(
        training_set_options,
        model_options,
        lr_options,
        loss_options,
        additive_smooth_options,
    ):
        experiment_name = f"{states_str}_{model}_{lr}_{loss}"

        output_dir = "../output_rep/hp_gridsearch_pittsburgh_with_prior_as_input"

        command = (
            "python train.py program.overwrite=True config_file=../conf/enviroatlas.yml"
            + f" experiment.name={experiment_name}"
            + f" experiment.module.segmentation_model={model}"
            + f" experiment.module.learning_rate={lr}"
            + f" experiment.module.loss={loss}"
            + f" experiment.module.num_filters=128"
            + f" experiment.datamodule.batch_size=128"
            + f" experiment.module.include_prior_as_datalayer=True"
            + f" experiment.datamodule.include_prior_as_datalayer=True"
            + f" experiment.module.output_smooth={additive_smooth}"
            + f" experiment.datamodule.states_str={states_str}"
            + f" experiment.datamodule.train_set={train_set}"
            + f" experiment.datamodule.val_set={val_set}"
            + f" experiment.datamodule.test_set={test_set}"
            + f" program.output_dir={output_dir}"
            + f" program.log_dir=../logs/hp_gridsearch_pittsburgh_with_prior_as_input"
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
