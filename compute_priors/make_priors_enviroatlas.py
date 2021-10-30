import numpy as np
import rasterio
import sys
import os
from importlib import reload
import time
import pandas as pd
import rasterio
import torch as T
from torchvision.transforms import GaussianBlur

sys.path.append("/home/esther/qr_for_landcover/scripts")
import landcover_definitions as lc

use_gpu = True
if use_gpu:
    device = T.device('cuda:0')
else:
    device = T.device('cpu')
    
sys.path.append('../scripts')
import util
from general_prior_calculation import calculate_prior_preblurred

from make_priors_chesapeake import compile_and_save_one_hot_nlcd_cc, compile_and_save_one_blurred_hot_nlcd_cc
    

cooc_matrix_dir = "/home/esther/qr_for_landcover/compute_priors/cooccurrence_matrices/enviroatlas"
ea_data_dir = '/home/esther/torchgeo_data/enviroatlas'
ea_data_splits_dir = '/home/esther/qr_for_landcover/enviroatlas_data_splits'


def condense_ea11_to_ea6_cooccurrance(cooc_matrix_in, keep_idxs =[0,1,2,3,4,6]):
    cooc = cooc_matrix_in.copy()
    # remap shrub to tree (only relevant for AZ)
    barren_idx = 3
    tree_idx = 4
    shrub_idx = 5
    cooc[tree_idx] += cooc[shrub_idx]
    # reindex and renormalize
    cooc = cooc[keep_idxs]
    cooc = (cooc / cooc.sum(axis=0))
    return cooc

def process_counts_per_state(state, tiles):
    counts_by_img, counts_renorm = make_cooccurance_matrices(tiles, state, save_results=False)
    counts_renorm_11 = counts_renorm.copy()
    counts_renorm_11[0,counts_renorm_11.sum(axis=0) == 0] = 1.0
    counts_renorm_6 = condense_ea11_to_ea6_cooccurrance(counts_renorm_11)

    
    return counts_renorm_6
  

        
def make_and_save_priors_ea(tiles_this, 
                            dir_this,
                            version, 
                            state, 
                            cooccurrence_mapping_matrices,
                            plot_mapping_matrix=False,
                            within_torchgeo=True):

    # by default use OSM
    ignore_osm = False
    ignore_buildings = False
    spatial_smooth = True
    
    #in EA5 it is this: (use CC 5 in case matrix or NLCD has nodatas)
    road_idx_cc = 2
    building_idx_cc = 2
    water_idx_cc = 1

       
    if version == 'from_cooccurrences_101_31':
        nlcd_blur_kernelsize = 101
        nlcd_blur_sigma = 31
        waterways_preblur_radius = 2
        water_bag_radius = 2
        buildings_bag_radius = 4
        roads_bag_radius = 4
        nlcd_mapping_matrix = cooccurrence_mapping_matrices[state]
        
    elif version == 'from_cooccurrences_101_31_no_osm_no_buildings':
        ignore_osm = True
        ignore_buildings = True
        nlcd_blur_kernelsize = 101
        nlcd_blur_sigma = 31
        waterways_preblur_radius = 2
        water_bag_radius = 2
        buildings_bag_radius = 4
        roads_bag_radius = 4
        nlcd_mapping_matrix = cooccurrence_mapping_matrices[state]
        
    else:
        print(f'version {version} not understood')
        return
    
    print(version)
    if spatial_smooth: print(nlcd_blur_sigma)
    else: print("not performing spatial smooth")
        
    if plot_mapping_matrix:
        plt.figure()
        plt.imshow(nlcd_mapping_matrix)
        plt.title('mapping matrix being used');
        plt.show()
        
    for i,tile_id in enumerate(tiles_this):
        print(i, tile_id)
        t1 = time.time()

        if spatial_smooth:
            if within_torchgeo:
                nlcd_fp = f'{dir_this}/{tile_id}_b_nlcd_onehot_blurred_kernelsize_' + \
                            f'{nlcd_blur_kernelsize}_sigma_{nlcd_blur_sigma}.tif'
            else:
                nlcd_fp = f'{dir_this}/{tile_id}/b3_nlcd_onehot_blurred_kernelsize_' + \
                            f'{nlcd_blur_kernelsize}_sigma_{nlcd_blur_sigma}.tif'
        else:
            nlcd_fp = f'{dir_this}/{tile_id}/b2_nlcd_onehot.tif'
        print(nlcd_fp)
        if within_torchgeo:
            roads_fp = f'{dir_this}/{tile_id}_c_roads.tif'
            buildings_fp = f'{dir_this}/{tile_id}_e_buildings.tif'
            waterways_fp = f'{dir_this}/{tile_id}_d1_waterways.tif'
            waterbodies_fp = f'{dir_this}/{tile_id}_d2_waterbodies.tif'   
        else:
            roads_fp = f'{dir_this}/{tile_id}/c_roads.tif'
            buildings_fp = f'{dir_this}/{tile_id}/e_buildings.tif'
            waterways_fp = f'{dir_this}/{tile_id}/d1_waterways.tif'
            waterbodies_fp = f'{dir_this}/{tile_id}/d2_waterbodies.tif'

        nlcd_preblurred = rasterio.open(nlcd_fp).read()
        
        if not ignore_buildings:
            buildings = rasterio.open(buildings_fp).read()[0]
        else:
            print('ignoring buildings')
            buildings = np.zeros(nlcd_preblurred.shape[1:])
            
        if not ignore_osm:
            roads = rasterio.open(roads_fp).read()[0]
            waterways = rasterio.open(waterways_fp).read()[0]
            waterbodies = rasterio.open(waterbodies_fp).read()[0]
            water = np.maximum(waterways, waterbodies)
            water = np.maximum(waterbodies, util.blur(waterways,waterways_preblur_radius)==1)

        # blur water
        if not ignore_osm:
            roads = util.bag_from_numpy(roads.astype('float'),radius=roads_bag_radius)
            water = util.bag_from_numpy(water.astype('float'),radius=water_bag_radius)
            
        if not ignore_buildings:
            buildings = util.bag_from_numpy(buildings.astype('float'),radius=buildings_bag_radius)
        
        if ignore_osm:
            print('ignoring roads, and water')
            roads = np.zeros(buildings.shape)
            water = np.zeros(buildings.shape)
            
        prior_this_6 = calculate_prior_preblurred(nlcd_preblurred,
                                                    roads,
                                                    buildings,
                                                    water,
                                                    nlcd_mapping_matrix,
                                                    road_idx = road_idx_cc,
                                                    building_idx = building_idx_cc,
                                                    water_idx = water_idx_cc)
        
        # now exclude nodata classes from the prior
        prior_this = prior_this_6[1:]
        
        # write this prior
        prof_out = rasterio.open(nlcd_fp).profile
        prof_out['count'] = prior_this.shape[0]
        
        if within_torchgeo:
            with rasterio.open(f'{dir_this}/{tile_id}_prior_{version}.tif', 'w', **prof_out) as f:
                f.write((prior_this*255.).astype(np.uint8))
        else:
            with rasterio.open(f'{dir_this}/{tile_id}/prior_{version}.tif', 'w', **prof_out) as f:
                f.write((prior_this*255.).astype(np.uint8))

        t2 = time.time()
        print(t2-t1, ' seconds')
        
if __name__ == "__main__":
    
    cities_all = ['pittsburgh_pa-2010',
                  'phoenix_az-2010', 
                  'austin_tx-2012', 
                  'durham_nc-2012'
    ]
    city_state_yr_pa = 'pittsburgh_pa-2010'
    test_cities = ['phoenix_az-2010','austin_tx-2012','durham_nc-2012']
    
    sets_pa = ['train', 'val']
    sets_all_cities = ['test', 'val5']
      
    tile_id_from_fn = lambda x: x.split('/')[-1][2:12]

    
    # compile the tile ids for each set
    tile_ids_by_state = {}
    for city_state_yr in cities_all:
        tile_ids_by_state[city_state_yr] = {}
        state = city_state_yr.split('-')[0][-2:]
        print(city_state_yr)
        print(state)
        fp_val5 = f'{ea_data_splits_dir}/{state}_val_5pts'
        fp_eval = f'{ea_data_splits_dir}/{state}_eval_10pts'
        tile_ids_by_state[city_state_yr]['val5'] = list(pd.read_csv(fp_val5)['img_fn'].apply(tile_id_from_fn))
        tile_ids_by_state[city_state_yr]['test'] = list(pd.read_csv(fp_eval)['img_fn'].apply(tile_id_from_fn))
    
    fp_pa_val = f'{ea_data_splits_dir}/pa_validation_8pts'
    fp_pa_train = f'{ea_data_splits_dir}/pa_train_10pts'
    tile_ids_by_state['pittsburgh_pa-2010']['train'] = list(pd.read_csv(fp_pa_train)['img_fn'].apply(tile_id_from_fn))
    tile_ids_by_state['pittsburgh_pa-2010']['val'] = list(pd.read_csv(fp_pa_val)['img_fn'].apply(tile_id_from_fn))
    
    cooccurrence_mapping_matrix_fns = {
        'pittsburgh_pa-2010': f'{cooc_matrix_dir}/avg_nlcd_pa_whole_city_label_cooccurrences.npy',
        'durham_nc-2012': f'{cooc_matrix_dir}/avg_nlcd_nc_whole_city_label_cooccurrences.npy',
        'phoenix_az-2010': f'{cooc_matrix_dir}/avg_nlcd_az_whole_city_no_ag_label_cooccurrences.npy',
        'austin_tx-2012': f'{cooc_matrix_dir}/avg_nlcd_austin_tx_whole_city_no_ag_label_cooccurrences.npy'
    }
    
    # 1. aggregate cooccurance matrices
    cooccurrence_mapping_matrices = {}
    for state in cities_all:
        fp_this_state = cooccurrence_mapping_matrix_fns[state]
        cooc_11_this_state = np.load(fp_this_state)
        cooccurrence_mapping_matrices[state] = condense_ea11_to_ea6_cooccurrance(cooc_11_this_state)

    # 2. make and store onehot NLCD and onhot blurred NLCD
    for state in cities_all:
        for set_this in sets_all_cities:

            dir_this_set = f'{ea_data_dir}/{state}_1m-{set_this}_tiles-debuffered'
            tiles_this_set = tile_ids_by_state[state][set_this]

            print(tiles_this_set)
            for i,tile_id in enumerate(tiles_this_set):
                print(i, tile_id)
                t1 = time.time()
                nlcd_fp = f'{dir_this_set}/{tile_id}_b_nlcd.tif'
                compile_and_save_one_hot_nlcd_cc(nlcd_fp,remap_to_idx = False) 
                t2 = time.time()
                print(t2-t1, ' seconds (onehot)')
                compile_and_save_one_blurred_hot_nlcd_cc(nlcd_fp.replace('nlcd','nlcd_onehot'),
                                                                      blur_kernel_size=101,
                                                                      blur_sigma=31,
                                                                      device=T.device('cuda:0')) 
                
    # do the additional PA sets
    for set_this in sets_pa:
        state = city_state_yr_pa
        dir_this_set = f'{ea_data_dir}/{state}_1m-{set_this}_tiles-debuffered'
        tiles_this_set = tile_ids_by_state[state][set_this]

        for i,tile_id in enumerate(tiles_this_set):
            print(i, tile_id)
            t1 = time.time()
            nlcd_fp = f'{dir_this_set}/{tile_id}_b_nlcd.tif'
            compile_and_save_one_hot_nlcd_cc(nlcd_fp,remap_to_idx = False) 
            t2 = time.time()
            print(t2-t1, ' seconds (onehot)')
            compile_and_save_one_blurred_hot_nlcd_cc(nlcd_fp.replace('nlcd','nlcd_onehot'),
                                                                      blur_kernel_size=101,
                                                                      blur_sigma=31,
                                                                      device=device) 
            
            
    # 3. fuse the priors
    # both with and without osm and buildings -- without is used for learning the prior
    prior_versions_todo = [
                           'from_cooccurrences_101_31_no_osm_no_buildings',
                           'from_cooccurrences_101_31'
                          ]

    for prior_version in prior_versions_todo:
        for state in cities_all:
            for set_this in sets_all_cities:

                dir_this_set = f'{ea_data_dir}/{state}_1m-{set_this}_tiles-debuffered'
                tiles_this_set = tile_ids_by_state[state][set_this]

                make_and_save_priors_ea(tiles_this_set, 
                                        dir_this_set, 
                                        version=prior_version,
                                        state=state,
                                        cooccurrence_mapping_matrices = cooccurrence_mapping_matrices,
                                        plot_mapping_matrix=False,
                                        within_torchgeo=True )
     
        # do the additional PA sets
        for set_this in sets_pa:
            state = city_state_yr_pa
            dir_this_set = f'{ea_data_dir}/{state}_1m-{set_this}_tiles-debuffered'
            tiles_this_set = tile_ids_by_state[state][set_this]

            make_and_save_priors_ea(tiles_this_set, 
                                        dir_this_set, 
                                        version=prior_version,
                                        state=state,
                                        cooccurrence_mapping_matrices = cooccurrence_mapping_matrices,
                                        plot_mapping_matrix=False,
                                        within_torchgeo=True )

            