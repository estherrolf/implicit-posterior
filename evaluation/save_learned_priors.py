import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

# local import
import run_model_forward_and_produce_tifs

# change this to wherever your training output is stored
torchgeo_output_dir = '../output_rep'
preds_save_dir = '../../torchgeo_predictions_rep'
torchgeo_data_dir = '/torchgeo_data'

# states to compute the learned prior for
states_to_eval = [
    "phoenix_az-2010_1m",
    "austin_tx-2012_1m",
    "durham_nc-2012_1m",
    "pittsburgh_pa-2010_1m",
]


loss_to_eval_options = ["nll"]

run_dirs = ["ea_learn_the_prior"]


for run_dir in run_dirs:
    for states_str in states_to_eval:

        if "pittsburgh" in states_str:
            # pittsburgh needs learned priors for the val set also in case you want to use
            # the val set to pick hps in the step where we learn _on_ the learned prior.

            sets_to_eval = ["test", "val"]
        else:
            sets_to_eval = ["test"]

        for loss in loss_to_eval_options:
            t1 = time.time()

            # load up the model
            run_name = (
                f"{states_str}_fcn_larger_0.0001_nll_blur_sigma_31_learn_the_prior"
            )
            model_kwargs = {
                "output_smooth": 1e-8,
                "classes": 5,
                "num_filters": 128,
                "in_channels": 9,
            }
            prior_version = "prior_from_cooccurrences_101_31_no_osm_no_buildings"

            ckpt_name = "last.ckpt"
            model_ckpt_fp = os.path.join(
                torchgeo_output_dir, run_dir, run_name, ckpt_name
            )

            for set_this in sets_to_eval:

                data_dir_this_state = f"{torchgeo_data_dir}/enviroatlas_lotp/{states_str}-{set_this}_tiles-debuffered"
                # these no osm priors are the input
                image_fns = [
                    os.path.join(data_dir_this_state, x)
                    for x in os.listdir(data_dir_this_state)
                    if x.endswith(f"{prior_version}.tif")
                ]

                # reorder the output names
                output_fns = [
                    x.replace(f"{prior_version}.tif", f"prior_learned_101_31.tif")
                    for x in image_fns
                ]

                extra_fns = []
                for img_fn in image_fns:
                    extra_fns_this_img = []
                    for data_type in [
                        "e_buildings",
                        "c_roads",
                        "d2_waterbodies",
                        "d1_waterways",
                    ]:
                        extra_fns_this_img.append(
                            img_fn.replace(f"{prior_version}.tif", f"{data_type}.tif")
                        )

                    extra_fns.append(extra_fns_this_img)

                # make all the output filepaths if they don't already exists
                if not os.path.exists(f"{preds_save_dir}"):
                    os.mkdir(f"{preds_save_dir}")
                    print(f"making dir {preds_save_dir}")
                if not os.path.exists(f"{preds_save_dir}/{run_name}"):
                    os.mkdir(f"{preds_save_dir}/{run_name}")
                    print(f"{preds_save_dir}/{run_name}")
                if not os.path.exists(f"{preds_save_dir}/{run_name}/enviroatlas_lotp"):
                    os.mkdir(f"{preds_save_dir}/{run_name}/enviroatlas_lotp")
                    print(f"making dir {preds_save_dir}/{run_name}/enviroatlas_lotp")
                if not os.path.exists(
                    f"{preds_save_dir}/{run_name}/enviroatlas_lotp/{states_str}-{set_this}_tiles-debuffered"
                ):
                    os.mkdir(
                        f"{preds_save_dir}/{run_name}/enviroatlas_lotp/{states_str}-{set_this}_tiles-debuffered"
                    )
                    print(
                        f"making dir {preds_save_dir}/{run_name}/enviroatlas_lotp/{states_str}-{set_this}_tiles-debuffered"
                    )

                print(model_ckpt_fp)
                
                model_type = 'fcn-larger'
                padding_larger_fcn = 10

                # run through tifs and save the output
                run_model_forward_and_produce_tifs.run_through_tiles(
                    model_ckpt_fp,
                    image_fns[:],
                    output_fns[:],
                    evaluating_learned_prior=True,
                    model=model_type,
                    gpu=0,
                    overwrite=True,
                    model_kwargs=model_kwargs,
                    include_prior_as_datalayer=True,
                    prior_fns=extra_fns,
                    edge_padding=padding_larger_fcn,
                )

                t2 = time.time()
                if set_this == "test":
                    print(f"{t2-t1} seconds for ten tiles")
                elif set_this == "val":
                    print(f"{t2-t1} seconds for eight tiles")
