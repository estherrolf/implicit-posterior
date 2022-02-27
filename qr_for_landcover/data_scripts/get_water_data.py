# Script downloads and saves rasterized water bodies (polygons) and waterways (lines) as from OSM
import time
import pickle
import rasterio
import rasterio.features
import fiona
import fiona.transform
import osmnx as ox
import networkx as nx
import numpy as np
import argparse 

def get_waterways(north, south, east, west):
    cf_waterway = '["waterway"]["tunnel"!~"culvert|yes"]'
    
    try:
        G_waterway = ox.graph_from_bbox(north, south, east,west, custom_filter=cf_waterway, retain_all=True)
    except:
        print('no waterway found matching description')
        G_waterway = None
    return G_waterway

def get_waterbodies(north,south,east,west):
    tags_water={"natural":"water"}
    geom_waterarea = ox.geometries_from_bbox(north, south, east,west, tags=tags_water)
    return geom_waterarea
    
def project_and_rasterize_waterways(G_waterway,
                                    src_crs,
                                    src_shape,
                                    src_transform,
                                    all_touched=True):
    
    # handle case where graph is empty or has no edges
    if G_waterway is None or len(G_waterway.edges()) == 0:
        return np.zeros(src_shape).astype(np.uint8)
    G_waterway_projected = ox.project_graph(G_waterway, to_crs=src_crs.to_string())
    
    edges = G_waterway_projected.edges(data=True)
    
    # first two indices of the edges are the nodes in the graph corresponding to the edge
    data_index = 2
    waterway_geometries = [e[data_index]['geometry'] for e in edges]
    # raterize
    waterways_raster = rasterio.features.rasterize(waterway_geometries, 
                                               out_shape=src_shape, 
                                               fill=0, 
                                               transform=src_transform, 
                                               all_touched=all_touched, 
                                               default_value=1, 
                                               dtype=np.uint8)    
    return waterways_raster

def project_and_rasterize_waterbodies(geometries_waterway,
                                    src_crs,
                                    src_shape,
                                    src_transform,
                                    all_touched=True):
    geometries_proj = [ox.projection.project_geometry(x, to_crs = src_crs.to_string())[0] for x in geometries_waterway]

    if len(geometries_proj) == 0:
        return np.zeros(src_shape).astype(np.uint8)
    
    waterbodies_raster = rasterio.features.rasterize(geometries_proj, 
                                                     out_shape=src_shape, 
                                                     fill=0, 
                                                     transform = src_transform, 
                                                     all_touched=all_touched, 
                                                     default_value=1, 
                                                     dtype=np.uint8)
    
    return waterbodies_raster


def download_water_one_tile(img_to_match_fn, tile_path_out, waterways=True, waterbodies=True):
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
    
    dst_profile = src_profile.copy()
    dst_profile["dtype"] = rasterio.uint8
    dst_profile["count"] = 1
    
    # first do waterways
    if waterways:
        G_waterway = get_waterways(north, south, east, west)
        waterways_raster = project_and_rasterize_waterways(G_waterway, 
                                                           src_crs, 
                                                           #src_data.shape[:2], 
                                                           src_data_shape,
                                                           src_transform,
                                                           all_touched=True)
        # save waterways tif
        f = rasterio.open(tile_path_out.replace('water','waterways'), "w", **dst_profile)
        f.write(waterways_raster.astype(np.uint8),1)
        f.close()
        
    # second do waterbodies
    if waterbodies:
        geom_waterarea = get_waterbodies(north, south, east, west)
        waterbodies_raster = project_and_rasterize_waterbodies(list(geom_waterarea['geometry']), 
                                                               src_crs, 
                                                               #src_data.shape[:2], 
                                                               src_data_shape,
                                                               src_transform,
                                                              all_touched=False)
        # save waterbodies tif
        f = rasterio.open(tile_path_out.replace('water','waterbodies'), "w", **dst_profile)
        f.write(waterbodies_raster.astype(np.uint8),1)
        f.close()
        
def download_water_full_block(state, 
                              block,
                              tile_path_out,
                              tile_folder= '/datadrive/data/tiles/'):

    # logic to download data and safe for each tile in the block -- same as in get_road_data.py
    #q2t = pickle.load(open(tile_path + tile_fn, 'rb'))
    tile_fn = f'{tile_folder}{state}_{block}_tiles'
    q2t = pickle.load(open(tile_fn, 'rb'))

    for i,t in enumerate(q2t):            
        start = time.time()
        
        if year is None:
            fn = q2t[t]['naip']
        else:
            # use this if you want to align to NAIP imagery of a certain year
            fn = q2t[t][str(year)]
            # do both waterways and waterbodies -- keep them separate so we can blur separately in the prior
            download_water_one_tile(fn, tile_path_out + t + '_water.tif', waterways=True, waterbodies=True)


        # record progress
        total_water_raster = np.maximum(waterbodies_raster,waterways_raster)
        print('Tile {} of {} {}: {}s'.format(i+1, len(q2t), t, time.time()-start))
        print('Water density:',total_water_raster.sum() / total_water_raster.shape[-1] / total_water_raster.shape[-2])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("block")
    parser.add_argument("state")
    args = parser.parse_args()
                              
    tile_path_out = f'/mnt/blobfuse/web-tool-data/lsr_change_detection/tiles/{args.state}/'
    
    download_water_full_block(args.state, args.block, tile_path_out)                
    
    
    