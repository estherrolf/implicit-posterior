# note: this is (slightly) adapted based on Caleb's notebook in the private land-cover-private repo:
# https://github.com/calebrob6/land-cover-private/blob/master/notebooks/Example%20-%20Reproject%20AI4E%20NLCD%20to%20NAIP%20tile%20boundaries.ipynb
# and follows the instructions at: 
# https://github.com/microsoft/AIforEarthDataSets/blob/users/dan/sentinel-2-nb-updates/data/nlcd.ipynb

import os
env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', 
           AWS_NO_SIGN_REQUEST='YES',
           GDAL_MAX_RAW_BLOCK_CACHE_SIZE='200000000',
           GDAL_SWATH_SIZE='200000000',
           VSI_CURL_CACHE_SIZE='200000000')
os.environ.update(env)

import numpy as np

import affine
import rasterio
import rasterio.enums
import rasterio.vrt

import sys
sys.path.append('../scripts')
import landcover_definitions as lc

# following instructions from 
# https://github.com/microsoft/AIforEarthDataSets/blob/users/dan/sentinel-2-nb-updates/data/nlcd.ipynb

account_name = 'cpdataeuwest'
container = 'cpdata'
year = 2016
area = 'conus'

nlcd_url = f'https://{account_name}.blob.core.windows.net/{container}/raw/nlcd/{area}/30m/{year}.tif'


def get_nlcd_copy_naip_extent(naip_fn, reindex=True, tile_path_out=None):
    # naip_fn can be a local fn or a url
    
    # get the parameters of the naip tif to match
    with rasterio.open(naip_fn) as f:
        naip_left = f.bounds.left
        naip_top = f.bounds.top
        naip_height = f.height
        naip_width = f.width
        naip_crs = f.crs.to_string()
        naip_profile = f.profile
     #   naip_data = np.rollaxis(f.read(), 0, 3)
        
        
    with rasterio.open(nlcd_url) as src:
    
        xres = 1.0 # meters
        yres = 1.0 # meters
        dst_transform = affine.Affine(xres, 0.0, naip_left, 0.0, -yres, naip_top)

        # copy the 
        vrt_options = {
            'resampling': rasterio.enums.Resampling.nearest,
            'crs': naip_crs,
            'transform': dst_transform,
            'height': naip_height,
            'width': naip_width,
        }

        with rasterio.vrt.WarpedVRT(src, **vrt_options) as vrt:
            nlcd_data = vrt.read().squeeze()
            nlcd_profile = vrt.profile
            
    # reindex so that values are contigous
    if reindex:
        nlcd_data = lc.map_raw_lc_to_idx['nlcd'][nlcd_data]
        
    if not tile_path_out is None:
        profile_out = naip_profile
        profile_out['count'] = 1
        with rasterio.open(tile_path_out,'w', **profile_out) as f:
            f.write(nlcd_data[np.newaxis,...])
        
            
    return nlcd_data, nlcd_profile