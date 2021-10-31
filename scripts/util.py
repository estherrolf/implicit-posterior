import numpy as np
import scipy.io
import rasterio
import matplotlib.pyplot as plt
import os
import torch as T



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



    

    
    