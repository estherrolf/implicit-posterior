import numpy as np
import scipy.io
import rasterio
import matplotlib.pyplot as plt
import os
import torch as T

# nlcd_cl = [ 0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95, 255 ]
# c2i = {cl:i for i,cl in enumerate(nlcd_cl)}


#blur function for waterays and roads
def blur(X_raster, bag_width=2):
    # blur once to expand width
    X  = X_raster.astype(np.float32)
    X_max = np.max(X)
    X += bag(T.from_numpy(X),bag_width).numpy()
    X[X>0] = 1.
    # blur again
    X += bag(T.from_numpy(X),bag_width).numpy()
    
    # cap at the origimal max
    X[X> X_max] = X_max
    return X

# copy this function from cluster-lsr/clustering.py
def bag_from_numpy(p, radius):
    return T.nn.functional.avg_pool2d(T.from_numpy(p).unsqueeze(0), 2*radius+1, stride=1, padding=radius, count_include_pad=False)[0].numpy()


def bag(p, radius):
    return T.nn.functional.avg_pool2d(p.unsqueeze(0), 2*radius+1, stride=1, padding=radius, count_include_pad=False)[0]

# def plot_grid(pred1, pred2, prior, img1, img2, nlcd, s=4, years = ['year1', 'year2'], save_fn = None): 
#     s = 4    
#     plt.figure(figsize=(15*s,10*s))
    
#     plt.subplot(231)
#     plt.title('land cover {}'.format(years[0] ))
#     plt.imshow(vis_nlcd(pred1).T.swapaxes(0,1))
#     plt.subplot(232)
#     plt.title('land cover {}'.format(years[1]))
#     plt.imshow(vis_nlcd(pred2).T.swapaxes(0,1))
#     plt.subplot(233)
#     plt.title('prior')
#     plt.imshow(vis_nlcd(prior).T.swapaxes(0,1))
#     plt.subplot(234)
#     plt.title(years[0])
#     plt.imshow(img1[:3].T.swapaxes(0,1))
#     plt.subplot(235)
#     plt.title(years[1])
#     plt.imshow(img2[:3].T.swapaxes(0,1))
#     plt.subplot(236)
#     plt.title('NLCD')
#     plt.imshow(vis_nlcd(nlcd,sparse=True).T.swapaxes(0,1))
#     #plt.show()
#     if not save_fn is None:
#         plt.savefig(save_fn)


# def latlon_to_tileid(lat, lon):
#     # if working in the us negate lon
#     lon = -lon
    
#     lat_rounded, lon_rounded = np.floor(lat).astype(int), np.floor(lon).astype(int)
#     lat_res, lon_res = lat - lat_rounded, lon - lon_rounded

#     # find block
#     block = '{0}{1:03n}'.format(lat_rounded, lon_rounded)
#     #print(block)

#     # find tile and corner within the block
#     # add 1 for each 1/8'th of latitude
#     lon_index = np.ceil((1-lon_res)*8.0).astype(int)
#     lon_corner = 'w' if (1-lon_res)*8.0 < (lon_index - 0.5) else 'e'

#     # note the long is flipped 
#     lat_index = np.ceil((1-lat_res)*8.0).astype(int)
#     lat_corner = 'n' if (1-lat_res)*8.0 < (lat_index - 0.5) else 's'

#     tile_index = lon_index + 8*(lat_index - 1)
#     corner =  lat_corner + lon_corner
#     tile_id = "{0}{1:02n}_{2}".format(block, tile_index, corner)
#     #print(tile_id)
#     #print(str(lon_index)+lon_corner, str(lat_index)+lat_corner)
#     return tile_id

# def tile_id_to_center_latlon(tileid):
#     lat = float(tileid[:2]) + 1
#     lon = -(float(tileid[2:5]) + 1)
    
#     # chunk_idxs are 1 indexed so subtract 1 first
#     chunk_id = int(tileid[5:7]) - 1
#     lat -= (chunk_id // 8)/8.0
#     lon += (chunk_id % 8)/8.0
    
#     if tileid[8] == 's':
#         lat -= 1/16.0
#     if tileid[9] == 'e':
#         lon += 1/16.0
        
#     # make sure it's in the center of the tile
#     lat -= 1/32.
#     lon += 1/ 32.
    
#     return lat, lon




def per_class_iou(y_true, y_pred, class_vals):
    """Calculate the IoUs for each class in class_vals."""
    
    # For each of the four classes compute the intersection and union 
    intersections = []
    unions = []
    ious = []
    for class_val in class_vals:
        mask_true = np.array(y_true == class_val)
        mask_pred = np.array(y_pred == class_val)

        intersection = np.logical_and(mask_pred,mask_true).sum()
        # todo change this to a max
        union = np.logical_or(mask_pred,mask_true).sum()

        intersections.append(intersection)
        unions.append(union)

        if union != 0:
            ious.append(intersection / union)
        else:
            ious.append(np.nan)
    return ious, intersections, unions

def aggregate_ious(ints_all, unions_all):
    """Aggregate IoUs across many image instances."""
    
    n_classes = len(ints_all[0])
    ints_summed = []
    unions_summed = []
    ious_summed = []
    
    for c in range(n_classes):
        intersection =np.sum([x[c] for x in ints_all])
        union = np.sum([x[c] for x in unions_all])              
        ints_summed.append(intersection)
        unions_summed.append(union)
        ious_summed.append(intersection/union)
        
    return ious_summed, ints_summed, unions_summed



    

    
    