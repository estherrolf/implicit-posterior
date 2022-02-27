import time
import pickle
from scipy.signal import correlate2d
import numpy as np
import rasterio
import rasterio.features
import fiona
import fiona.transform
import osmnx as ox
import argparse


def download_roads_one_tile(img_to_match_fn, tile_path_out):
        # get geographic data from the naip filename
        with rasterio.open(img_to_match_fn, "r") as f:
            src_crs = f.crs
            src_bounds = f.bounds
            src_data = np.rollaxis(f.read(), 0, 3)
            src_data_shape = f.shape
            src_transform = f.transform
            src_profile = f.profile

        lons, lats = fiona.transform.transform(src_crs.to_string(), 
                                               "epsg:4326", 
                                               [src_bounds.left, src_bounds.right], 
                                               [src_bounds.top, src_bounds.bottom])

        eps=0.05
        north, south, east, west = lats[0]+eps, lats[1]-eps, lons[1]+eps, lons[0]-eps
        
        # network_type selects the type of road features pulled from openstreetmap
        G = ox.graph_from_bbox(north, south, east, west, network_type='all_private', retain_all=True) 
        G_projected = ox.project_graph(G, to_crs=src_crs.to_string())
        edges = G_projected.edges(data=True)
        # convert to shapes
        data_index = 2
        shapes = [e[data_index]['geometry'] for e in edges]
        # if there are any roads in this tile, convert them too raster format
        print('pre rasterize')
        if len(shapes) == 0:
            output = np.zeros(src_data_shape)
            print('no roads found; raster is all 0s')
        else:
            output = rasterio.features.rasterize(shapes,
                                                 out_shape=src_data_shape, 
                                                 fill=0, 
                                                 transform=src_transform, 
                                                 all_touched=False, 
                                                 default_value=1, 
                                                 dtype=np.uint8)

        # widen the roads a little
        kernel = np.ones((3,3))
        new_output = correlate2d(output, kernel, mode="same", boundary="fill", fillvalue=0)
        new_output[new_output>0] = 1
        new_output = new_output.astype(np.uint8)

        # save
        dst_profile = src_profile.copy()
        dst_profile["dtype"] = rasterio.uint8
        dst_profile["count"] = 1

        f = rasterio.open(tile_path_out, "w", **dst_profile)
        f.write(new_output, 1)
        f.close()

def download_roads_full_block(state, 
                              block,
                              tile_path_out,
                              year=None,
                              tile_folder= '/datadrive/data/tiles/'):
    
    tile_fn = f'{tile_folder}{state}_{block}_tiles'
    q2t = pickle.load(open(tile_fn, 'rb'))

    # loop through all tiles in the block
    for i,t in enumerate(q2t):
        
        start = time.time()
        if year is None:
            fn = q2t[t]['naip']
        else:
            # use this if you want to align to NAIP imagery of a certain year
            fn = q2t[t][str(year)]
        
        download_roads_one_tile(fn, tile_path_out + t + '_roads.tif')

        print('Tile {} of {} {}: {}s'.format(i+1, len(q2t), t, time.time()-start))
        print('Road density:',new_output.sum() / new_output.shape[-1] / new_output.shape[-2])
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("block")
    parser.add_argument("state")
    args = parser.parse_args()
                              
    tile_path_out = f'/mnt/blobfuse/web-tool-data/lsr_change_detection/tiles/{args.state}/'
    
    download_roads_full_block(args.state, args.block, tile_path_out)                              
                             
