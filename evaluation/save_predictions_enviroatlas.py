import rasterio
import numpy as np
import torch
import time
import os
import sys
import run_model_forward_and_produce_tifs 


torchgeo_output_dir = '/home/esther/qr_for_landcover/output_rep'
torchgeo_data_dir = '/datadrive/esther/torchgeo_data'
torchgeo_pred_dir = '/home/esther/torchgeo_predictions_rep'

states_to_eval = [
    'phoenix_az-2010_1m',
    'austin_tx-2012_1m',
    'durham_nc-2012_1m', 
    'pittsburgh_pa-2010_1m'
]
                       
                       
prior_version = 'from_cooccurrences_101_31'

run_dirs = [
 #   'hp_gridsearch_pittsburgh',
 #   'hp_gridsearch_pittsburgh_with_prior_as_input',
  #  'ea_qr_from_pittsburgh_model_rep',
#    'ea_qr_from_scratch_rep',
    'ea_qr_learned_prior'
]

loss_to_eval_by_run = {
    'hp_gridsearch_pittsburgh': ['nll'],
    'hp_gridsearch_pittsburgh_with_prior_as_input': ['nll'],
    'ea_qr_from_pittsburgh_model_rep': ['qr_forward', 'qr_reverse'],
    'ea_qr_from_scratch_rep': ['qr_forward', 'qr_reverse'],
    'ea_qr_learned_prior': ['qr_reverse']
}

for run_dir in run_dirs:
    for states_str in states_to_eval:
        for loss in loss_to_eval_by_run[run_dir]:
            t1 = time.time()

            include_prior_as_datalayer=False
            
            # qr model initialized with pittsburgh highres model
            if run_dir == 'ea_qr_from_pittsburgh_model_rep':
                if loss == 'qr_forward':
                    run_name = f'pa_checkpoint_{states_str}_fcn_1e-05_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'
                elif loss == 'qr_reverse':
                    run_name = f'pa_checkpoint_{states_str}_fcn_0.001_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'
                model_kwargs = {'output_smooth':1e-4, 'classes': 5, 'num_filters':128, 'in_channels': 4}
            
            # qr model initialized from random
            elif run_dir == 'ea_qr_from_scratch_rep':
                if loss == 'qr_forward':
                    run_name = f'{states_str}_fcn_1e-05_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'
                elif loss == 'qr_reverse':
                    run_name = f'{states_str}_fcn_0.001_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'

                model_kwargs = {'output_smooth':1e-4, 'classes': 5, 'num_filters':128, 'in_channels': 4}

            # hr model take the best from the grid search runs
            elif run_dir == 'hp_gridsearch_pittsburgh':    
                run_name = f'pittsburgh_pa-2010_1m_fcn_0.001_{loss}/'
                model_kwargs = {'output_smooth':1e-8, 'classes': 5, 'num_filters':128, 'in_channels': 4}
            
            # hr model on (naip + prior) take the best from the grid search runs
            elif run_dir == 'hp_gridsearch_pittsburgh_with_prior_as_input':    
                run_name = f'pittsburgh_pa-2010_1m_fcn_0.001_{loss}_with_prior/'
                include_prior_as_datalayer=True
                prior_type = 'prior_from_cooccurrences_101_31'
                model_kwargs = {'output_smooth':1e-8, 'classes': 5, 'num_filters':128, 'in_channels': 9}
                
            # qr model trained with the learned prior
            elif run_dir == 'ea_qr_learned_prior':
                prior_version = 'learned_101_31'
                if loss == 'qr_forward':
                    run_name = f'pa_checkpoint_{states_str}_fcn_1e-05_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'
                if loss == 'qr_reverse':
                    run_name = f'pa_checkpoint_{states_str}_fcn_0.001_{loss}_{prior_version}_additive_smooth_0.0001_prior_smooth_0.0001/'
                model_kwargs = {'output_smooth':1e-4, 'classes': 5, 'num_filters':128, 'in_channels': 4}

            ckpt_name = 'last.ckpt'
            model_ckpt_fp = os.path.join(torchgeo_output_dir,run_dir,run_name, ckpt_name)

            data_dir_this_state = f'{torchgeo_data_dir}/enviroatlas_lotp/{states_str}-test_tiles-debuffered'
            image_fns = [os.path.join(data_dir_this_state,x) for x in os.listdir(data_dir_this_state) if x.endswith('a_naip.tif')]

            # reorder the output names
            output_fns = [x.replace('a_naip.tif',f'{loss}_pred_last.tif') for x in image_fns]
            output_fns = [x.replace(f'{torchgeo_data_dir}/', f'{torchgeo_pred_dir}/{run_name}') for x in output_fns]

            # prior fns only matter if they're used as model input
            if include_prior_as_datalayer:
                prior_fns = [[x.replace('a_naip',f'prior_{prior_version}')] for x in image_fns]
            else: 
                prior_fns = ['' for x in image_fns]

            # make all the output filepaths if they don't already exists
            if not os.path.exists(f'{torchgeo_pred_dir}'):
                os.mkdir(f'{torchgeo_pred_dir}')
                print(f'making dir {torchgeo_pred_dir}')
            if not os.path.exists(f'{torchgeo_pred_dir}/{run_name}'):
                os.mkdir(f'{torchgeo_pred_dir}/{run_name}')
                print(f'making dir {torchgeo_pred_dir}/{run_name}')
            if not os.path.exists(f'{torchgeo_pred_dir}/{run_name}/enviroatlas_lotp'):
                os.mkdir(f'{torchgeo_pred_dir}/{run_name}/enviroatlas_lotp')
                print(f'making dir {torchgeo_pred_dir}/{run_name}/enviroatlas_lotp')
            if not os.path.exists(f'{torchgeo_pred_dir}/{run_name}/enviroatlas_lotp/{states_str}-test_tiles-debuffered'):
                os.mkdir(f'{torchgeo_pred_dir}/{run_name}/enviroatlas_lotp/{states_str}-test_tiles-debuffered')
                print(f'making dir {torchgeo_pred_dir}/{run_name}/enviroatlas_lotp/{states_str}-test_tiles-debuffered')

            print(model_ckpt_fp)
            
            # run through tifs and save the output
            run_model_forward_and_produce_tifs.run_through_tiles(model_ckpt_fp,
                                                                  image_fns[:],
                                                                  output_fns[:],
                                                                  gpu = 3,
                                                                  overwrite=True,
                                                                  model_kwargs=model_kwargs,
                                                                  include_prior_as_datalayer=include_prior_as_datalayer,
                                                                  prior_fns=prior_fns
                                                                 )

            t2 = time.time()
            print(f'{t2-t1} seconds for ten tiles')