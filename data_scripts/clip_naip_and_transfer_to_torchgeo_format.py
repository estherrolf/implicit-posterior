import math
import os

import numpy as np

import rasterio
import rasterio.mask

import fiona
import fiona.transform
import shapely.geometry
import shutil
import itertools
import enviroatlas_geographic_extrap_functions as ea
import pandas as pd
from importlib import reload


ea_dir_orig = '/home/esther/lc-mapping/enviroatlas_data'
ea_dir_torchgeo = '/home/esther/torchgeo_data/enviroatlas'
ea_data_splits_dir = '/home/esther/qr_for_landcover/enviroatlas_data_splits'


def fill_in_dir(state, city_state_year, this_set_name, tile_ids_this_set):
    """Transfer files from the original dir to the torchgeo data dir."""
    file_set_name = f'{city_state_yr}_1m'

    copy_this_dir = os.path.join(data_root, f'{file_set_name}-{this_set_name}_tiles')
    # if this dir doesn't exist make it
    if not os.path.exists(copy_this_dir):
        os.mkdir(copy_this_dir)
        print(f'making dir {copy_this_dir}')

    fns_to_copy = [
                    "a_naip.tif",
                    "b_nlcd.tif",
                    "c_roads.tif",
                    "d_water.tif",
                    "d1_waterways.tif",
                    "d2_waterbodies.tif",
                    "e_buildings.tif",
                    "h_highres_labels.tif",
                   ]

    for tile_id in tile_ids_this_set:
        print(tile_id)
        for fn in fns_to_copy:
            shutil.copy2(f'{ea_dir_orig}/{state}/{tile_id}/{fn}',
                         os.path.join(copy_this_dir, tile_id + f"_{fn}"))
            
            
            
def do_debuffer(input_fn, output_fn, amount=600):
    """ Caleb's function to clip the naip tiles by amount pixels per side."""
    with rasterio.open(input_fn) as src:
        box = shapely.geometry.box(*src.bounds)
        box = box.buffer(-amount)
        geom = shapely.geometry.mapping(box)        
        profile = src.profile.copy()
       
        shape_mask, transform, window = rasterio.mask.raster_geometry_mask(
            src, [geom], all_touched=False, invert=False, crop=True,
            pad=False, pad_width=0.5
        )
        out_shape = (src.count, ) + shape_mask.shape
        data = src.read(
            window=window, out_shape=out_shape, masked=False, indexes=None
        )

    num_channels, height, width = data.shape
        
    profile["width"] = width
    profile["height"] = height
    profile["transform"] = transform
    profile["blockxsize"] = 512
    profile["blockysize"] = 512
    profile["tiled"] = True
    profile["compress"] = "deflate"
    profile["interleave"] = "pixel"

    with rasterio.open(output_fn, "w", **profile) as f:
        f.write(data)
        
if __name__ == "__main__":

    # Transfer all the files to the new dataset
    # Loop over train, val in pittsburgh
    # and val5, test in all cities
    cities_all = ['pittsburgh_pa-2010',
                  'phoenix_az-2010', 
                  'austin_tx-2012', 
                  'durham_nc-2012']
    city_state_yr_pa = 'pittsburgh_pa-2010'
    test_cities = ['phoenix_az-2010','austin_tx-2012','durham_nc-2012']
    sets_pa = ['train', 'val']
    sets_all_cities = ['test', 'val5']
    
    # compile the tile ids for each set
    tile_ids = {}
    for city_state_yr in cities_all
        tile_ids[city_state_yr_pa] = {}
        state = city_state_yr.split('-')[0][-2:]
        tile_ids[city_state_yr_pa]['val5'] = pd.read_csv(f'{ea_data_splits_dir}/{state}_val_5pts')
        tile_ids[city_state_yr_pa]['test'] = pd.read_csv(f'{ea_data_splits_dir}/{state}_eval_10pts')
    
    tile_ids['pittsburgh_pa-2010']['train'] = pd.read_csv(f'{ea_data_splits_dir}/pa_validation_8pts')
    tile_ids['pittsburgh_pa-2010']['val'] = pd.read_csv(f'{ea_data_splits_dir}/pa_train_10pts')
        
    
    # do test and val5 sets in all cities
    for this_set_name, city_state_yr in itertools.product(sets_all_cities, cities_all):
                                                                             
        tile_ids_this_set = tile_ids[city_state_yr][this_set_name] # TODO index into dict
        print(city_state_yr, this_set_name)
        state = city_state_yr.split('-')[0][-2:]
        if state == 'tx': state = 'austin_tx'
        fill_in_dir(state, city_state_yr, this_set_name, tile_ids_this_set)
    
    # do train and val in pittsburgh
    for this_set_name in sets_pa:
        city_state_yr = city_state_yr_pa                                                                     
        tile_ids_this_set = tile_ids[city_state_yr][this_set_name] # TODO index into dict
        print(city_state_yr, this_set_name)
        state = city_state_yr.split('-')[0][-2:]
        if state == 'tx': state = 'austin_tx'
        fill_in_dir(state, city_state_yr, this_set_name, tile_ids_this_set)
    
   
    
    # debuffer/clip all the files
    datasets_val5 = [f"{city_state_yr}_1m-val5_tiles" for city_state_yr in cities_all] 
    datasets_test = [f"{city_state_yr}_1m-test_tiles" for city_state_yr in cities_all]
    datsets_pa = [f"{city_state_yr_pa}_1m-{set_this}_tiles" for set_this in sets_pa]
    datasets_all =  datasets_pa + datasets_test + datasets_val5
    
    for i, dataset in enumerate(datasets):
        print(f"{i}/{len(datasets)} -- {dataset}")
        fns = [
            fn
            for fn in os.listdir(f"{ea_dir_torchgeo}/{dataset}/")
        ]
        new_dataset = dataset.replace("_tiles","_tiles-debuffered")
        os.makedirs(f"{ea_dir_torchgeo}/{new_dataset}/", exist_ok=True)
        for j, fn in enumerate(fns):
            print(f"{j}/{len(fns)}")
            do_debuffer(
                f"{ea_dir_torchgeo}/{dataset}/{fn}",
                f"{ea_dir_torchgeo}/{new_dataset}/{fn}"
            )
        
        
    
    # create the spatial indeex
    geoms = []
    splits = []
    naip_fns = []
    for i, dataset in enumerate(datasets_all):
        print(f"{i}/{len(datasets)} -- {dataset}")
        new_dataset = dataset.replace("_tiles","_tiles-debuffered")
        fns = [
            fn
            for fn in os.listdir(f"{ea_dir_torchgeo}/{new_dataset}/")
            if fn.endswith("a_naip.tif")
        ]
        for j, fn in enumerate(fns):
            print(f"{j}/{len(fns)}")
            with rasterio.open(f"{ea_dir_torchgeo}/{new_dataset}/{fn}") as f:
                src_crs = f.crs.to_string()        
                box = shapely.geometry.box(*f.bounds)
                geom = shapely.geometry.mapping(box)
                geom = fiona.transform.transform_geom(src_crs, "epsg:3857", geom)
                geoms.append(geom)
                splits.append(dataset)
            naip_fns.append(f"{new_dataset}/{fn}")
            
            
    schema = {
    "geometry": "Polygon",
    "properties": {
                    "split": "str",
                    "a_naip": "str",
                    "b_nlcd": "str",
                    "c_roads": "str",
                    "d_water": "str",
                    "d1_waterways": "str",
                    "d2_waterbodies": "str",
                    "e_buildings": "str",
                    "h_highres_labels": "str",
        }
    }

    with fiona.open(f"{ea_dir_torchgeo}/spatial_index.geojson", 
                    "w", driver="GeoJSON", crs="EPSG:3857", schema=schema) as f:
        
        for i in range(len(geoms)):
            split = splits[i]
            split = split.replace("_1m_2013_extended","")
            split = split.replace("_1m_2014_extended","")
            split = split.replace("_tiles","")
            row = {
                "type": "Feature",
                "geometry": geoms[i],
                "properties": {
                    "split": splits[i][:-6],
                    "a_naip": naip_fns[i],
                    "b_nlcd": naip_fns[i].replace("a_naip.tif","b_nlcd.tif"),
                    "c_roads": naip_fns[i].replace("a_naip.tif","c_roads.tif"),
                    "d_water": naip_fns[i].replace("a_naip.tif","d_water.tif"),
                    "d1_waterways": naip_fns[i].replace("a_naip.tif","d1_waterways.tif"),
                    "d2_waterbodies": naip_fns[i].replace("a_naip.tif","d2_waterbodies.tif"),
                    "e_buildings": naip_fns[i].replace("a_naip.tif","e_buildings.tif"),
                    "h_highres_labels": naip_fns[i].replace("a_naip.tif","h_highres_labels.tif"),
            }
            f.write(row)
        