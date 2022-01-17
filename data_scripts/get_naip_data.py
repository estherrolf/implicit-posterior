import os
import rasterio
from azure.storage.blob import ContainerClient

env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', 
           AWS_NO_SIGN_REQUEST='YES',
           GDAL_MAX_RAW_BLOCK_CACHE_SIZE='200000000',
           GDAL_SWATH_SIZE='200000000',
           VSI_CURL_CACHE_SIZE='200000000')
os.environ.update(env)


def get_img_fns_azure(state, naip_year, tile_ids=None):
    # return all image paths on azure for a given state and year
    # if tile_ids is specified return only paths matching those tile ids
    
    storage_account_name = 'naipblobs'
    container_name = 'naip'
    storage_account_url = 'https://' + storage_account_name + '.blob.core.windows.net/'

    container_client = ContainerClient(account_url=storage_account_url, 
                                                     container_name=container_name)
    azure_scene_prefix = f"v002/{state}/{naip_year}"
    generator = container_client.list_blobs(name_starts_with=azure_scene_prefix)

    if tile_ids is None:
        blob_names = [blob.name for blob in generator if blob.name.endswith('.tif')]
    
    else:
        blob_names =  [blob.name for blob in generator if blob.name.split('/')[-1][2:12] in tile_ids \
                                                            and blob.name.endswith('.tif')]
        
    container_client.close()
    image_paths = [f'{storage_account_url}{container_name}/{x}' for x in blob_names]
    
    return image_paths

def get_naip_fn(tile_id, state, naip_year):
    image_fns = get_img_fns_azure(state,naip_year,[tile_id])
    
    # there should just be one fn returned since we're just asking for one tile_id
    if len(image_fns) != 1:
        print(f'we expected 1 fn and we found {len(image_fns)}:')
        print(image_fns)
        return
    
    return image_fns[0]

