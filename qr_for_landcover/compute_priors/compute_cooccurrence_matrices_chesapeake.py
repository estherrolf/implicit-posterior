import os
import sys
import numpy as np
import rasterio
import sklearn.preprocessing

sys.path.append("/home/esther/qr_for_landcover/scripts")
import landcover_definitions as lc

# number of classes for each LC classification scheme
num_cs_classes = len(lc.class_definitions["chesapeake_7"])
num_nlcd_classes = len(lc.class_definitions["nlcd"])

# data dir in which the Chesapeake labels van be found
CC_data_dir = "/home/esther/torchgeo_data/cvpr_chesapeake_landcover"
cooc_ouput_dir = "/home/esther/qr_for_landcover/compute_priors/cooccurrence_matrices"


def unique_tile_ids_from_dir(data_dir_this):
    """Get tile ids from a provided data dir."""
    return np.unique([x[2:17] for x in os.listdir(data_dir_this)])


def get_cooccurrence_counts(tile_ids_loop, dir_loop):
    """Loop through tile ids to get per-image cooccurrence matrices."""
    counts_by_img = np.zeros(
        (len(tile_ids_loop), num_cs_classes, num_nlcd_classes), dtype=int
    )
    
    print(f"of  {len(tile_ids_loop)}: ", end = "")
    for i, tile_id in enumerate(tile_ids_loop):
        print(f"{i} ", end = ""); sys.stdout.flush()

        hr_labels = rasterio.open(f"{dir_loop}/m_{tile_id}_lc.tif").read()[0]
        nlcd = rasterio.open(f"{dir_loop}/m_{tile_id}_nlcd.tif").read()[0]
        
        # handle weird case of 128 being nodata in WV
        nlcd[nlcd == 128] = 0
        
        nlcd_reindexed = lc.map_raw_lc_to_idx["nlcd"][nlcd].astype(np.uint8)
        hr_reindexed = hr_labels.astype(np.uint8)

        counts_this_img = lc.count_cooccurances(
            nlcd_reindexed, hr_reindexed, "nlcd", "chesapeake_7"
        )

        counts_by_img[i] = counts_this_img
    print()
    return counts_by_img


def normalize_counts_by_image(counts_by_img):
    """Convert per-image cooccurrence matrices to normalized averages."""
    # renormalize the counts matrix
    all_zero_mask = counts_by_img[:, 1:].sum(axis=(1, 2)) == 0
    counts_avgd_nonzero = counts_by_img[~all_zero_mask].mean(axis=0)
    counts_renorm = sklearn.preprocessing.normalize(
        counts_avgd_nonzero, norm="l1", axis=0
    )

    # if theres' no occurances of any NLCD class, map that NLCD class to nodata
    counts_renorm[0][counts_renorm.sum(axis=0) == 0] = 1.0
    return counts_renorm


def make_and_save_cooccurrences_by_state(state, year):
    """Loop over states in the Chesapeake Conservancy dataset."""
    # get the data dirs
    dir_train = f"{CC_data_dir}/{state}_1m_{year}_extended-debuffered-train_tiles"
    tiles_train = unique_tile_ids_from_dir(dir_train)

    counts_by_img = get_cooccurrence_counts(tiles_train, dir_train)
    counts_renorm = normalize_counts_by_image(counts_by_img)

    # save the big cooccurence matrix
    print(f"{cooc_ouput_dir}/chesapeake/nlcd_{state}_train_label_cooccurrences.npy")
    np.save(
        f"{cooc_ouput_dir}/chesapeake/nlcd_{state}_train_label_cooccurrences.npy",
        counts_by_img,
    )
    np.save(
        f"{cooc_ouput_dir}/chesapeake/avg_nlcd_{state}_train_label_cooccurrences.npy",
        counts_renorm,
    )


if __name__ == "__main__":
    states = ["de", "md", "ny", "pa", "va", "wv"]
    years = [2013, 2013, 2013, 2013, 2014, 2014]

    for state, year in zip(states, years):
        print(f"computing cooccurrence matrix for {state} - {year}")

        make_and_save_cooccurrences_by_state(state, year)
