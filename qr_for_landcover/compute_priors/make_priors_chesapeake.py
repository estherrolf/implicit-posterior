import numpy as np
import rasterio
import sys
import os
from importlib import reload
import time
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

CC_data_dir = "/home/esther/torchgeo_data/cvpr_chesapeake_landcover"
cooc_data_dir = "/home/esther/qr_for_landcover/compute_priors/cooccurrence_matrices/chesapeake"

def unique_tile_ids_from_dir(data_dir_this):
    """Get tile ids from a provided data dir."""
    return np.unique([x[2:17] for x in os.listdir(data_dir_this)])

def condense_cc7_to_cc5_cooccurrance(cooc_matrix_in):
    """Combine Impervious classes."""
    cooc = cooc_matrix_in.copy()
    cooc[4] = cooc[4:].sum(axis=0)
    return cooc[:5]


def compile_and_save_one_hot_nlcd_cc(nlcd_fp, remap_to_idx=True):
    if remap_to_idx: 
        nlcd = lc.map_raw_lc_to_idx['nlcd'][rasterio.open(nlcd_fp).read()[0]]
    else:
        nlcd = rasterio.open(nlcd_fp).read()[0]
        
    num_classes = len(lc.lc_colors['nlcd'])
    nlcd_onehot = np.zeros((num_classes, nlcd.shape[0],nlcd.shape[1]))
    for c in range(num_classes):
        nlcd_onehot[c][nlcd == c] = 1

    prof_out = rasterio.open(nlcd_fp).profile.copy()
    prof_out['count'] = num_classes
    
    nlcd_fp_out = nlcd_fp.replace('nlcd', 'nlcd_onehot')
    
    with rasterio.open(nlcd_fp_out, 'w', **prof_out) as f:
        f.write(nlcd_onehot)
        

def compile_and_save_one_blurred_hot_nlcd_cc(nlcd_onehot_fp,
                                          blur_kernel_size=101,
                                          blur_sigma=51,
                                          device=T.device('cuda:0')):
    
    nlcd_onehot = rasterio.open(nlcd_onehot_fp).read()    
    nlcd_onehot = T.tensor(nlcd_onehot, device=device).float()
    
    # guassian blur
    blur = GaussianBlur(blur_kernel_size, blur_sigma)
    nlcd_onehot_blurred = blur(nlcd_onehot).cpu().numpy()
    

    prof_out = rasterio.open(nlcd_onehot_fp).profile.copy()
    
    nlcd_fp_out = nlcd_onehot_fp.replace('nlcd_onehot', 
                                         f'nlcd_onehot_blurred_kernelsize_{blur_kernel_size}_sigma_{blur_sigma}')
    with rasterio.open(nlcd_fp_out, 'w', **prof_out) as f:
        f.write((nlcd_onehot_blurred * 255.).astype(np.uint8))
        
def make_and_save_priors_cc(tiles_this, 
                            dir_this,
                            version, 
                            state, 
                            cooccurrence_mapping_matrices,
                            plot_mapping_matrix=False):

    # by default use OSM
    ignore_osm = False
    ignore_buildings = False
    spatial_smooth = True
    
    #in CC5 it is this: (use CC 5 in case matrix or NLCD has nodatas)
    road_idx_cc = 4
    building_idx_cc = 4
    water_idx_cc = 1
        
    if version == 'from_cooccurrences_101_31_no_osm_no_buildings':
        nlcd_blur_kernelsize = 101
        nlcd_blur_sigma = 31
        nlcd_mapping_matrix = cooccurrence_mapping_matrices[state]
        ignore_osm = True
        ignore_buildings = True
            
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
            # note this assumes that you've made this blurred input already.
            nlcd_fp = f'{dir_this}/m_{tile_id}_nlcd_onehot_blurred_kernelsize_' + \
                            f'{nlcd_blur_kernelsize}_sigma_{nlcd_blur_sigma}.tif'
            
        else:
            nlcd_fp = f'{dir_this}/m_{tile_id}_nlcd_onehot.tif'
        print(nlcd_fp)
        
        #by default these will get ignored
        roads_fp = f'{dir_this}/m_{tile_id}_roads.tif'
        buildings_fp = f'{dir_this}/m_{tile_id}_buildings.tif'
        waterways_fp = f'{dir_this}/m_{tile_id}_waterways.tif'
        waterbodies_fp = f'{dir_this}/m_{tile_id}_waterbodies.tif'

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
            
        prior_this_5 = calculate_prior_preblurred(nlcd_preblurred,
                                                roads,
                                                buildings,
                                                water,
                                                nlcd_mapping_matrix,
                                                road_idx = road_idx_cc,
                                                building_idx = building_idx_cc,
                                                water_idx = water_idx_cc)
        
        # now exclude nodata classes from the prior
        prior_this = prior_this_5[1:]
        
        # write this prior
        prof_out = rasterio.open(nlcd_fp).profile
        prof_out['count'] = prior_this.shape[0]
        with rasterio.open(f'{dir_this}/m_{tile_id}_prior_{version}.tif', 'w', **prof_out) as f:
            f.write((prior_this*255.).astype(np.uint8))

        t2 = time.time()
        print(t2-t1, ' seconds')
        
     
if __name__ == "__main__":
    
    states = ["de", "md", "ny", "pa", "va", "wv"]
    years = [2013, 2013, 2013, 2013, 2014, 2014]
    sets = ['val', 'test', 'train']
    
    cooc_matrices = {}
    for state in states: 
        fp_this_state = f'{cooc_data_dir}/avg_nlcd_{state}_train_label_cooccurrences.npy'
        cooc_7_this_state = np.load(fp_this_state)
        cooc_matrices[state] = condense_cc7_to_cc5_cooccurrance(cooc_7_this_state) 
    
    for set_this in sets:
        for state, year in zip(states,years):        
            print(f"computing cooccurrence matrix for {state} - {set_this}")
            # where to find this data
            dir_this_set = f'{CC_data_dir}/{state}_1m_{year}_extended-debuffered-{set_this}_tiles'
            tiles_this_set = unique_tile_ids_from_dir(dir_this_set)
            
            # 1. save the onehot NLCD and blurred onehot_nlcd
            print('saving onehot and blured onehot nlcd')
            print(f"of  {len(tiles_this_set)}: ", end = "")
            for i,tile_id in enumerate(tiles_this_set):
                print(f"{i} ", end = ""); sys.stdout.flush()
                nlcd_fp = f'{dir_this_set}/m_{tile_id}_nlcd.tif'
                
                # save onhhot nlcd
                compile_and_save_one_hot_nlcd_cc(nlcd_fp) 

                # save blurred nlcd
                compile_and_save_one_blurred_hot_nlcd_cc(nlcd_fp.replace('nlcd','nlcd_onehot'),
                                                                      blur_kernel_size=101,
                                                                      blur_sigma=31,
                                                                      device=device) 
                
                
     
            # 2. make the priors
            make_and_save_priors_cc(tiles_this_set,  
                                    dir_this_set,
                                    'from_cooccurrences_101_31_no_osm_no_buildings', 
                                    state, 
                                    cooc_matrices,
                                    plot_mapping_matrix=False)

        