import numpy as np
import torch as T
from torchvision.transforms import GaussianBlur

def blurred_nlcd_to_prior_soft(nlcd_blurred, mapping_matrix, ensure_normalized = True):
    """Creates a "soft" or reindexed nlcd image, depending on the mapping matrix.

    e.g. if mapping matrix is the identity, you'll map a sparce LC into one
    with one dimension for each nlcd class
    if the mapping matrix has nonzero entries, the output array will represent
    a probability or "soft" mapping over output classes
    """
    num_output_classes, num_orig_classes = mapping_matrix.shape
    lc_out = np.zeros((num_output_classes,)+nlcd_blurred.shape[1:])
    if ensure_normalized:
        mapping_matrix = mapping_matrix / mapping_matrix.sum(axis=0)
    
    for c_orig in range(num_orig_classes):
        for c_out in range(num_output_classes):
            lc_out[c_out,:] += mapping_matrix[c_out, c_orig] * nlcd_blurred[c_orig]
     
    return lc_out


def nlcd_to_prior_soft(nlcd_orig, mapping_matrix, ensure_normalized = True):
    """Creates a "soft" or reindexed nlcd image, depending on the mapping matrix.

    e.g. if mapping matrix is the identity, you'll map a sparce LC into one
    with one dimension for each nlcd class
    if the mapping matrix has nonzero entries, the output array will represent
    a probability or "soft" mapping over output classes
    """
    num_output_classes, num_orig_classes = mapping_matrix.shape
    lc_out = np.zeros((num_output_classes,)+nlcd_orig.shape)
    
    if ensure_normalized:
        mapping_matrix = mapping_matrix / mapping_matrix.sum(axis=0)
    
    for c_orig in range(num_orig_classes):
      #  print(c_orig)
        x_idxs, y_idxs = np.where(nlcd_orig == c_orig)
      #  print(x_idxs)
        if len(x_idxs) > 0:
            for c_out in range(num_output_classes):
                lc_out[c_out,x_idxs,y_idxs] = mapping_matrix[c_out, c_orig]
     
    return lc_out


def calculate_prior_preblurred(nlcd_blurred, 
                                roads, 
                                buildings, 
                                water,         
                                nlcd_mapping_matrix,
                                road_idx,
                                building_idx,
                                water_idx):
    """Fuse input data into a prior, where NLCD input is blurred."""
    
    # nlcd comes as "hard" classes, so translate them to a soft prior
    prior = blurred_nlcd_to_prior_soft(nlcd_blurred.astype(float),nlcd_mapping_matrix)
    # normalize before adding in the
    prior = T.nn.functional.normalize(T.tensor(prior),p=1,dim=0).cpu().numpy()
    
    # add in these features -- this will give them half weight in the total sum
    prior[water_idx] += water.astype('float')
    prior[road_idx] += roads.astype('float')
    prior[building_idx] += buildings.astype('float')
    
    # normalize then release from GPU
    prior =  T.nn.functional.normalize(T.tensor(prior),p=1,dim=0).cpu().numpy()
    T.cuda.empty_cache()
    
    return prior
    
    
    