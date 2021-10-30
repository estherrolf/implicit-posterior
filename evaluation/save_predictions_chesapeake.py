import rasterio
import numpy as np
import torch
import time
import os
import sys

import run_model_forward_and_produce_tifs 

torchgeo_output_dir = '/home/esther/torchgeo/output'
torchgeo_data_dir = '/home/esther/torchgeo_data'
torchgeo_pred_dir = '/home/esther/torchgeo_predictions'


states_to_eval = ['ny', 'pa', 'ny+pa']
                       
loss_to_eval_options = ['qr_forward', 'qr_reverse']
prior_version = 'from_cooccurrences_101_31_no_osm_no_buildings'

include_prior_as_datalayer=False
    
run_dirs = ['chesepeake_north_qr']

# output smooth is 1e-4 for all of the runs
model_kwargs = {'output_smooth':1e-4, 'classes': 4, 'num_filters': 128, 'in_channels': 4}            

for run_dir in run_dirs:
    for states_str in states_to_eval:
        for loss in loss_to_eval_options:
            t1 = time.time()

            run_name = f'{states_str}_fcn_0.0001_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'

            ckpt_name = 'last.ckpt'
            model_ckpt_fp = os.path.join(torchgeo_output_dir,run_dir,run_name, ckpt_name)

            states = states_str.split('+')
            
            img_fns = []
            
            # eval for each state to keep them separate
            for state in states:
                state_identifier = f'{state}_1m_2013_extended-debuffered-test_tiles'
                data_dir_this_state = f'{torchgeo_data_dir}/cvpr_chesapeake_landcover/{state_identifier}'
            
                image_fns = [os.path.join(data_dir_this_state,x) for x in os.listdir(data_dir_this_state) if x.endswith('naip-new.tif')]

                print(len(image_fns))

                # reorder the output names
                output_fns = [x.replace('naip-new.tif',f'{loss}_pred_last.tif') for x in image_fns]
                output_fns = [x.replace(torchgeo_data_dir, f'{torchgeo_pred_dir}/{run_name}') for x in output_fns]

                print(len(image_fns))

                # prior fns only matter if they're used as model input
                if include_prior_as_datalayer:
                     prior_fns = [[x.replace('a_naip',f'prior_{prior_version}')] for x in image_fns]
                else: 
                    prior_fns = [[''] for x in image_fns]

                # make all the output filepaths if they don't already exists
                if not os.path.exists(f'{torchgeo_pred_dir}'):
                    os.mkdir(f'{torchgeo_pred_dir}')
                    print(f'making dir {torchgeo_pred_dir}')
                if not os.path.exists(f'{torchgeo_pred_dir}/{run_name}'):
                    os.mkdir(f'{torchgeo_pred_dir}/{run_name}')
                    print(f'making dir {torchgeo_pred_dir}/{run_name}')
                if not os.path.exists(f'{torchgeo_pred_dir}/{run_name}/cvpr_chesapeake_landcover'):
                    os.mkdir(f'{torchgeo_pred_dir}/{run_name}/cvpr_chesapeake_landcover')
                    print(f'making dir {torchgeo_pred_dir}/{run_name}/cvpr_chesapeake_landcover')
                if not os.path.exists(f'{torchgeo_pred_dir}/{run_name}/cvpr_chesapeake_landcover/{state_identifier}'):
                    os.mkdir(f'{torchgeo_pred_dir}/{run_name}/cvpr_chesapeake_landcover/{state_identifier}')
                    print(f'making dir {torchgeo_pred_dir}/{run_name}/cvpr_chesapeake_landcover/{state_identifier}')

                print(model_ckpt_fp)

                # run through tifs and save the output
                run_model_forward_and_produce_tifs.run_through_tiles(model_ckpt_fp,
                                                                      image_fns[:],
                                                                      output_fns[:],
                                                                      gpu = 0,
                                                                      overwrite=True,
                                                                      model_kwargs=model_kwargs,
                                                                      include_prior_as_datalayer=include_prior_as_datalayer,
                                                                      prior_fns=prior_fns
                                                                     )

            t2 = time.time()
            print(f'{t2-t1} seconds for ten tiles')