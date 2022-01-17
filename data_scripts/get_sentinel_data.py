from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
import planetary_computer as pc

import shapely
import rasterio
from rasterio import windows
from rasterio import features
from rasterio import warp


import requests
import subprocess
import os 
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../scripts')
from img_util import reproject_and_crop

def get_sentinel_10m_stac(area_of_interest, timeframe, reference_tif, output_dir='tmp/',
        cloud_cover_idx=0, verbose=False):
    '''Returns 10m-resolution sentinel color data from the Planetary Computer.

    Args:
        area_of_interest (dict): GeoJSON-formatted polygon that specifies the area of interest.
        timeframe (str):  Time frame to search images in formatted as "YYYY-MM-DD/YYY-MM-DD1" (from-to).
        reference_tif (str): Path to reference GeoTIFF to reproject to.
        output_dir (str): Directory to save outputs.
        cloud_cover_idx (int): Cloud cover index of returned imagery (0 is least cloud cover).
        verbose (bool): Verbosity flag.
    Returns:
        filename_reprojected (str): Path to reprojected Sentinel raster.
    '''
    # Search for imagery in given area-time
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=timeframe,
        query={"eo:cloud_cover": {"lt": 80}},
    )
    items = list(search.get_items())
    if verbose:
        print(f"Returned {len(items)} Items")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    # Sort results by cloud cover
    cloud_cover_sorted = sorted(items, key=lambda item: eo.ext(item).cloud_cover)

    # Get RGB URI
    asset_href = cloud_cover_sorted[cloud_cover_idx].assets['visual-10m'].href
    signed_href = pc.sign(asset_href)

    # Download RGB
    rgb_filename = output_dir + asset_href.split('/')[-1]
    response = requests.get(signed_href)
    with open(rgb_filename, 'wb') as f:
        f.write(response.content)

    # Get NIR URI
    asset_href = cloud_cover_sorted[cloud_cover_idx].assets['B08'].href
    signed_href = pc.sign(asset_href)

    # Download NIR
    nir_filename = output_dir + asset_href.split('/')[-1]
    response = requests.get(signed_href)
    with open(nir_filename, 'wb') as f:
        f.write(response.content)

    # Split RGB into 3 separate TIFFs
    for i, color in enumerate(['_red.tif', '_green.tif', '_blue.tif']):
        split_color_command = [
            "gdal_translate",
            "-b", str(i+1),
            rgb_filename,
            rgb_filename.split('.')[0] + color
        ]
        subprocess.call(split_color_command)

    # Merge all TIFFs
    filename = rgb_filename.split('.')[0] + '_B08.tif'
    vrt_command = [
        "gdalbuildvrt",
        "-separate",
        filename.split('.')[0] + ".vrt",
        rgb_filename.split('.')[0] + '_red.tif',
        rgb_filename.split('.')[0] + '_green.tif',
        rgb_filename.split('.')[0] + '_blue.tif',
        nir_filename
    ]
    subprocess.call(vrt_command)
    merge_command = [
        "gdal_translate",
        filename.split('.')[0] + ".vrt",
        filename
    ]
    subprocess.call(merge_command)

    # Clean up temp files
    clean_command = [
        "rm",
        filename.split('.')[0] + ".vrt",
        rgb_filename.split('.')[0] + '_red.tif',
        rgb_filename.split('.')[0] + '_green.tif',
        rgb_filename.split('.')[0] + '_blue.tif'
    ]
    subprocess.call(clean_command)

    # Reproject raster to reference GeoTIFF
    filename_reprojected = filename.split('.')[0] + '_reprojected.tif'
    reproject_and_crop(filename, reference_tif, filename_reprojected, resample="near")
    return filename_reprojected



def get_sentinel_10m_stac_not_aligned(area_of_interest, timeframe, img_to_match=None, cloud_cover_idx=0, verbose=False):
    # Search for imagery in given area-time
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=timeframe,
        query={"eo:cloud_cover": {"lt": 80}},
    )
    items = list(search.get_items())
    if verbose:
        print(f"Returned {len(items)} Items")

    # Sort results by cloud cover
    cloud_cover_sorted = sorted(items, key=lambda item: eo.ext(item).cloud_cover)
    
    # Get URI
    asset_href = cloud_cover_sorted[cloud_cover_idx].assets['visual-10m'].href
    signed_href = pc.sign(asset_href)

    # Retrieve imagery
    with rasterio.open(signed_href) as ds:
        # Crop to area of interest
        aoi_bounds = features.bounds(area_of_interest)
        warped_aoi_bounds = warp.transform_bounds('epsg:4326', ds.crs, *aoi_bounds)
        aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
        img = ds.read(window=aoi_window)

    return img
                                
def get_sentinel_10m_fp_least_cloud_by_tileid(img_to_match, time_range):
    # convert the tile id to bounding box in sentinel CRS
    sentinel_crs = 'WGS84'
    
    with rasterio.open(img_to_match) as f:
        bounds_to_match = f.bounds
        crs_to_match = f.crs
    
    
    bounds_latlon = rasterio.warp.transform_bounds(crs_to_match, sentinel_crs, *bounds_to_match)
    
    bbox_coords = list(zip(list((shapely.geometry.box(*bounds_latlon).exterior.xy)[0]),
                           list((shapely.geometry.box(*bounds_latlon).exterior.xy)[1])))
    
    aoi = {'type':'Polygon',
      'coordinates': [bbox_coords]}
    
    return get_sentinel_10m_stac_fp(aoi, time_range, img_to_match)
    
    
def get_sentinel_10m_stac_fp(area_of_interest, timeframe, img_to_match=None, cloud_cover_idx=0, verbose=False):
    # Search for imagery in given area-time
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=timeframe,
        query={"eo:cloud_cover": {"lt": 80}},
    )
    items = list(search.get_items())
    if verbose:
        print(f"Returned {len(items)} Items")

    # Sort results by cloud cover
    cloud_cover_sorted = sorted(items, key=lambda item: eo.ext(item).cloud_cover)
    
    # Get URI
    asset_href = cloud_cover_sorted[cloud_cover_idx].assets['visual-10m'].href
    signed_href = pc.sign(asset_href)

    return signed_href
