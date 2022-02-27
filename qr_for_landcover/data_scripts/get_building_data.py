import os
import numpy as np
import rasterio
from get_naip_data import get_naip_fn

# if the buildings directory format changes or the data gets updated we need to update there
building_years = ['2015','2014','2013','2012','2011','2010']
buildings_base_url = 'https://mslandcoverstorageeast.blob.core.windows.net/resampled-buildings/data/v1'


states_by_year_cached ={
  #  '2015': ['az'],
    '2014': 'ca  nm'.split('  '),
 #   '2014': 'ca  nm tx'.split('  '),
    '2013': 'fl  la  md  me'.split('  '),
    '2012': 'ca  ct  in  ks  ky  ma  mi  mo  ms  nc  nd  ne  or  ri  sd  tn  tx  va  wy'.split('  '),
    '2011': 'al  co  de  ia  id  il  md  me  mt  nh  nm  ny  oh  or  sc  ut  va  vt  wa  wv'.split('  '),
    '2010': 'ar  az  fl  ga  la  mn  nj  nv  ok  pa  wi'.split('  ')
}
    
def year_by_state_cached(state):
    building_yrs = list(states_by_year_cached.keys())
    building_yrs.sort()
    # serach in reverse chronological order
    for yr in building_yrs[::-1]:
        if state in states_by_year_cached[yr]:
            return str(yr)



def year_by_state(state,building_yrs = building_years, buildings_path_base = buildings_base_url):
    building_yrs.sort()
    # serach in reverse chronological order
    for yr in building_yrs[::-1]:
        if state in os.listdir(f'{buildings_path_base}/{yr}/states'):
            return str(yr)
    


def get_building_filename(tile_id, 
                          state,
                          buildings_path_base = buildings_base_url,
                          buildings_saved_locally=False):
    
    # automatically figure out what the building id should be given what is saved
    if buildings_saved_locally:
        building_year = year_by_state(state, buildings_path_base = buildings_path_base)
    else:
        building_year = year_by_state_cached(state)
     
    block_id = tile_id.split('_')[0][:-2]

    f = f'{buildings_path_base}/{building_year}/states/{state}/{state}_1m_{building_year}/{block_id}/'
    if buildings_saved_locally:
        return np.sort([f'{f}/{x.name}' for x in os.scandir(f) if x.name.startswith("m_"+tile_id)])[0]
    
    else:
        # find the image_id of the naip tile from the building year
        naip_fn_building = get_naip_fn(tile_id,state,building_year).split('/')[-1]
        building_fn = naip_fn_building.replace('.tif','_building.tif')
        return f'{f}{building_fn}'
        
                    
def load_buildings(tile_id, 
                   state, 
                   buildings_path_base= buildings_base_url):
            
    block_id = tile_id.split('_')[0][:-2]
    if buildings_saved_locally:
        building_year = year_by_state(state, buildings_path_base = buildings_path_base)
    else:
        building_year = year_by_state_cached(state)
        
    buildings_path = f'{buildings_path_base}/v1/{building_year}/states/{state}/{state}_1m_{building_year}/{block_id}/'
    building_tilepath = buildings_path + get_building_filename(building_year, tile_id, state) 
    buildings = 1 - rasterio.open(building_tilepath).read()
    
    return buildings