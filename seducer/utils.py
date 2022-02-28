import numpy as np
import torch
import torch.nn.functional as F

'''
Labels:
    0 - Sky
    1 - Boat
    2 - Sea
'''
seducer_labels = [0, 1, 2]

# Artistic palette
seducer_label_colors_1 = {
    0:  ( 80,184,231), 
    1:  (255,255,255),
    2:  ( 32, 73, 92),
}

# Dull palette
seducer_label_colors_2 = {
    0:  (255,  0,  0), 
    1:  (  0,255,  0),
    2:  (  0,  0,255),
}

def labels_to_color(labels, categorical=True, label_idxs=seducer_labels, label_colors=seducer_label_colors_2):
    '''
    Transforms labels image to color image. 
    
    Args:
        labels (numpy array): Categorical/one-hot labels image (1,height,width)/(n_classes,height,width)
        categorical (bool): Whether the labels image is categorical or one-hot encoded
        label_idxs (list): Integers to use as labels
        label_colors (dict): Colors corresponding to integer labels
    Returns:
        out_img (np.array): Color image corresponding to predicted classes (3,height,width)
    '''
    # Create zeros image and add class colors sequentially
    out_img = np.zeros((3,) + labels.shape[1:])
    
    # Fill with label colors
    if categorical:
        for c in label_idxs:
            color = np.array(label_colors[c])[:, np.newaxis, np.newaxis]
            out_img += (labels == c).astype(int) * color
        # Cast to int and transpose to (height,width,channels)
        out_img = out_img.astype(int).transpose([1,2,0])
    else:
        for c in label_idxs:
            color = np.array(label_colors[c])[:, np.newaxis, np.newaxis]
            out_img += labels[c] * color
        # Cast to int and transpose to (height,width,channels)
        out_img = out_img.transpose([1,2,0]).astype(int)
    
    return out_img

def sample_from_img(img, prior, batch_size, side):
    '''Sample random patches from image and prior.
    
    Args:
        img (numpy array): Image to sample patches from (C,H,W)
        prior (numpy array): Prior to sample patches from (N,H,W)
        batch_size (int): Number of samples per batch to return
        side (int): Side of square patch to return
    Returns:
        img_batch (torch.Tesnor): Image patch tensor list
        prior_batch (torch.Tesnor): Prior patch tensor list
        pos_grid_batch (torch.Tensor): Position grid patch tensor list
    '''
    img_batch = []
    prior_batch = []
    pos_grid_batch = []
    
    for i in range(batch_size):
        # Select random patches from image and prior
        row = torch.randint(0, img.shape[1] - side, (1,)).item()
        col = torch.randint(0, img.shape[2] - side, (1,)).item()

        img_patch = img[:, row:row+side, col:col+side]
        prior_patch = prior[:, row:row+side, col:col+side]
        
        # Create positional encoding
        pos_grid_row = torch.tile(torch.arange(row, row+side, 1), (1, side, 1))
        pos_grid_row = torch.transpose(pos_grid_row, 1, 2)
        pos_grid_col = torch.tile(torch.arange(col, col+side, 1), (1, side, 1))
                                      
        pos_grid_row = pos_grid_row / (img.shape[1] - 1)
        pos_grid_col = pos_grid_col / (img.shape[2] - 1)
        
        pos_grid = torch.cat((pos_grid_row, pos_grid_col), dim=0)

        # Add to sample list
        img_batch.append(img_patch)
        prior_batch.append(prior_patch)
        pos_grid_batch.append(pos_grid)
        
    # Convert to batch tensors
    img_batch = torch.stack(img_batch, dim=0)
    prior_batch = torch.stack(prior_batch, dim=0)
    pos_grid_batch = torch.stack(pos_grid_batch, dim=0)
    
    return img_batch, prior_batch, pos_grid_batch

def run_on_tile_full(net, tile):
    '''Run CNN on full tile and return resulting predictions.
    
    Args:
        net (torch.nn.Module): Network to use for predicting
        tile (torch.Tensor): Tile to run network on
    Returns:
        pred_tile (np.array): Predicted probabilities array (n_classes,height,width)
    '''
    with torch.no_grad():
        # Create position grid
        pos_grid_row = torch.tile(torch.arange(0, tile.shape[-2], 1), (1, tile.shape[-1], 1))
        pos_grid_row = torch.transpose(pos_grid_row, 1, 2)
        pos_grid_col = torch.tile(torch.arange(0, tile.shape[-1], 1), (1, tile.shape[-2], 1))
                                      
        pos_grid_row = pos_grid_row / tile.shape[-2]
        pos_grid_col = pos_grid_col / tile.shape[-1]
        
        pos_grid = torch.cat((pos_grid_row, pos_grid_col), dim=0)
        
        # Pass image/position grid through network
        tile_in = tile.unsqueeze(0).float().to(net.get_device())
        pos_grid_in = pos_grid.unsqueeze(0).float().to(net.get_device())
        pred_tile = net(tile_in, pos_grid_in)
        
    return pred_tile.squeeze(0).cpu().numpy()

def loss_on_prior(logged_output, target):
    '''QR (forward) loss on prior.
    
    Args:
        logged_output (torch.Tensor): Log-probabilities of predictions
        target (torch.Tensor): Prior probabilities
    Returns:
        loss (torch.Tensor): Computed loss
    '''
    q = torch.exp(logged_output)
    q_mean = torch.mean(q, dim=(0,2,3), keepdims=True)
    loss = (torch.sum(q_mean * torch.log(q_mean))
            - torch.einsum('bcxy,bcxy->bxy', q, torch.log(target)).mean())
    
    return loss

def loss_on_prior_reverse_kl(logged_output, target):
    '''RQ (backwards) loss on prior.
    
    Args:
        logged_output (torch.Tensor): Log-probabilities of predictions
        target (torch.Tensor): Prior probabilities
    Returns:
        loss (torch.Tensor): Computed loss
    '''
    q = torch.exp(logged_output)
    z = F.normalize(q, p=1, dim=(0,2,3))
    r = F.normalize(z * target, p=1, dim=1)
        
    loss = torch.einsum('bcxy,bcxy->bxy', r, torch.log(r) - torch.log(q)).mean() 
    return loss
