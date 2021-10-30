import numpy as np
import pandas as pd
import os
import sys
import rasterio
import time
import sklearn.preprocessing

sys.path.append("/home/esther/qr_for_landcover/scripts")
import landcover_definitions as lc

sys.path.append("/home/esther/qr_for_landcover/data_scripts")
import get_naip_data, get_building_data, get_nlcd_data, get_road_data, get_water_data

# path to where the city csvs exists
enviroatlas_city_csvs_path = '/home/esther/land-cover-private/data/splits2'
ea_data_dir = '/home/esther/lc-mapping/enviroatlas_data'

# where to save cooccurrence matrices
cooc_ouput_dir = "/home/esther/qr_for_landcover/compute_priors/cooccurrence_matrices/enviroatlas"


num_nlcd_classes = len(lc.class_definitions['nlcd'])
num_ea_classes = len(lc.class_definitions['enviroatlas'])

def annotate_splits(splits_df):
    """ Process splits and add columns describing fraction of zeros in the labels."""
    
    # add percent zeros row for the labels
    pct_zeros = []
    for i in range(len(splits_df)):
        l = rasterio.open(splits_df.iloc[i]['label_fn']).read()[0]
        pct_zeros.append((l == 0).mean())
    pct_zeros = np.array(pct_zeros)
    
    splits_df['frac_zeros'] = pct_zeros
    splits_df['all_zeros'] = pct_zeros == 1.0
    splits_df['some_zeros'] = pct_zeros > 0.0
    
    return splits_df


def add_annotation_to_splits(splits_csv):
    splits_this = pd.read_csv(splits_csv)
    splits_annotated = annotate_splits(splits_this)
    splits_annotated.to_csv(splits_csv.replace('.csv', '_annotated.csv'),index=None)
    

def make_cooccurance_matrices(splits, 
                              state, 
                              descripror,
                              lc_type = 'enviroatlas', 
                              reindex_hr=True, 
                              reindex_nlcd=False, 
                              save_results=True,
                              nodata_idx=0):
    """Compute cooccurrence matrices ."""
    num_images = len(splits)

    num_ea_classes = len(lc.class_definitions[lc_type])
    num_nlcd_classes = len(lc.class_definitions['nlcd'])

    counts_by_img = np.zeros((num_images, num_ea_classes, num_nlcd_classes),dtype=int)

    for i, row_instance in splits.iterrows():
        print(i)
        tile_id = row_instance['label_fn'].split('/')[-1][2:12]

        # assumes you've already processed each cell in the splits
        hr_labels = rasterio.open(f'{ea_data_dir}/{state}/{tile_id}/h_highres_labels.tif').read()[0]
        nlcd = rasterio.open(f'{ea_data_dir}/{state}/{tile_id}/b_nlcd.tif').read()[0]
        if reindex_nlcd:
            nlcd_reindexed = lc.map_raw_lc_to_idx['nlcd'][nlcd.astype(np.uint8)]
        else:
            nlcd_reindexed = nlcd.astype(np.uint8)
        
        if reindex_hr:
            hr_reindexed = lc.map_raw_lc_to_idx[lc_type][hr_labels].astype(np.uint8)
        
        else: hr_reindexed = hr_labels.astype(np.uint8)
        t1 = time.time()
        counts_this_img = lc.count_cooccurances(nlcd_reindexed, hr_reindexed, 'nlcd', lc_type)
        t2 = time.time()
       # print(f'{t2-t1:.2f} seconds counting')

        counts_by_img[i] = counts_this_img
    
    # aggregate
    all_zero_mask = counts_by_img[:,1:].sum(axis=(1,2)) == 0

    counts_avgd_nonzero = counts_by_img[~all_zero_mask].mean(axis=0)
    
    counts_renorm = sklearn.preprocessing.normalize(counts_avgd_nonzero,norm='l1',axis=0)

    # if a column is all missing put it as zero
    counts_avgd_nonzero[0] = 0
    # any fully zero columns put as nodata
    counts_avgd_nonzero[nodata_idx,counts_avgd_nonzero.sum(axis=0) == 0] = 1.
    counts_avgd_nonzero = sklearn.preprocessing.normalize(counts_avgd_nonzero,norm='l1',axis=0)
        
    # save the counts and averaged cooccurence matrices
    if save_results:
        np.save(f'{cooc_ouput_dir}/nlcd_{state}_{descriptor}_label_cooccurrences.npy', counts_by_img)
        np.save(f'{cooc_ouput_dir}/avg_nlcd_{state}_{descriptor}_label_cooccurrences.npy', counts_avgd_nonzero)
    
    return counts_by_img, counts_renorm

    
    
if __name__ == "__main__":
    
    splits_to_process = ['enviroatlas-austin_tx-2012.csv',
                         'enviroatlas-durham_nc-2012.csv',
                         'enviroatlas-phoenix_az-2010.csv',
                         'enviroatlas-pittsburgh_pa-2010.csv']


   # 1. add zero annotations
    for x in splits_to_process:
        print(x)
        annot_path = f'{enviroatlas_city_csvs_path}/{x}'.replace('.csv', '_annotated.csv')
        if not os.path.exists(annot_path):
            add_annotation_to_splits(f'{enviroatlas_city_csvs_path}/{x}')

        else:
            print(annot_path, ' already exists')
            
        # make a version with no zeros
        splits_all = pd.read_csv(annot_path)
        splits_nozeros = splits_all[splits_all['some_zeros']==False]
        splits_nozeros = splits_nozeros.reset_index(drop=True)
        splits_nozeros.to_csv(annot_path.replace('_annot', "_no_zeros_annot"))
        
        # exclude any tiles with agriculture instances from austin and phoenix
        if 'austin' in x or 'phoenix' in x:
            ag_idx = 7

            num_images = len(splits_nozeros)
            counts_by_img = np.zeros((num_images, num_ea_classes),dtype=int)
    
            for i,row in splits_nozeros.iterrows():
                print(i)
                lc_this = lc.map_raw_lc_to_idx['enviroatlas'][rasterio.open(row['label_fn']).read()]
                counts_this = np.array([(lc_this == x).sum() for x in range(num_ea_classes)])
                counts_by_img[i] = counts_this
                
                splits_nozeros_no_ag = splits_nozeros[counts_by_img[:,ag_idx] == 0].reset_index(drop=True)
                splits_nozeros_no_ag.to_csv(annot_path.replace("annotated", 
                                                               "annotated_no_zeros_no_agriculture"),
                                            index='None')
                
    # 2. compute and save the cooccurrance matrices
    
    states_todo = ['pa', 'austin_tx', 'az', 'nc']
    filenames_todo = [
                       f'{enviroatlas_city_csvs_path}/enviroatlas-pittsburgh_pa-2010_annotated.csv',
                      f'{enviroatlas_city_csvs_path}/enviroatlas-austin_tx-2012_annotated_no_zeros_no_agriculture.csv',
                      f'{enviroatlas_city_csvs_path}/enviroatlas-phoenix_az-2010_annotated_no_zeros_no_agriculture.csv',
                      f'{enviroatlas_city_csvs_path}/enviroatlas-durham_nc-2012_annotated.csv',
                     ]

    descriptors = [
                    'whole_city', 
                    'whole_city_no_ag', 
                    'whole_city_no_ag', 
                    'whole_city', 
                  ]
    
    for state, fn, descriptor in zip(states_todo, filenames_todo, descriptors):
        print(state, fn)

        splits = pd.read_csv(fn)

        make_cooccurance_matrices(splits, state, 
                                  descriptor,
                                  lc_type = 'enviroatlas', 
                                  reindex_hr=True, 
                                  reindex_nlcd=False, 
                                  save_results=True,
                                  nodata_idx=0)

                
    
            