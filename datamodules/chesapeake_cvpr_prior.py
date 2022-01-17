import sys
from typing import Any, Callable, Dict, List, Optional, Sequence
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

sys.path.append('/home/esther/torchgeo')
from torchgeo.datasets import ChesapeakeCVPR, ChesapeakeCVPRDataModule
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers.single import GridGeoSampler

class ChesapeakeCVPRPriorDataModule(ChesapeakeCVPRDataModule):
    """LightningDataModule implementation for the Chesapeake CVPR Land Cover dataset.
    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """

    def __init__(
        self,
        root_dir: str,
        train_splits: List[str],
        val_splits: List[str],
        test_splits: List[str],
        patches_per_tile: int = 200,
        patch_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 0,
        prior_smoothing_constant: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Chesapeake CVPR based DataLoaders.
        Args:
            root_dir: The ``root`` arugment to pass to the ChesapeakeCVPR Dataset
                classes
            train_splits: The splits used to train the model, e.g. ["ny-train"]
            val_splits: The splits used to validate the model, e.g. ["ny-val"]
            test_splits: The splits used to test the model, e.g. ["ny-test"]
            patches_per_tile: The number of patches per tile to sample
            patch_size: The size of each patch in pixels (test patches will be 1.5 times
                this size)
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            class_set: The high-resolution land cover class set to use - 5 or 7
            use_prior_labels: Flag for using a prior over high-resolution classes
                instead of the high-resolution labels themselves
            prior_smoothing_constant: additive smoothing to add when using prior labels
        Raises:
            ValueError: if ``use_prior_labels`` is used with ``class_set==7``
        """
        print(train_splits)
        super().__init__(root_dir, train_splits, val_splits, test_splits, patches_per_tile, patch_size, batch_size, num_workers, 5, True, prior_smoothing_constant)
        self.layers = [
            "naip-new",
            "lc",
            "prior_from_cooccurrences_101_31_no_osm_no_buildings",
        ]
        self.original_patch_size = int(patch_size * 4.0)
        if patch_size == 64: self.original_patch_size = int(patch_size * 6.0) 
        
    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesses a single sample.
        Args:
            sample: sample dictionary containing image and mask
        Returns:
            preprocessed sample
        """
        
        nodata_idx_pad_orig = 7
        nodata_idx_pad_condensed = 5 # note that we're about to subtract one
        
            
        sample["image"] = sample["image"] / 255.0
        sample["mask"] = sample["mask"].squeeze()
        
        high_res_labels = sample["mask"][0]
        prior = sample["mask"][1:]

        prior = F.normalize(prior.float(), p=1, dim=0)
        prior = F.normalize(
            prior + self.prior_smoothing_constant, p=1, dim=0
        )
            
        high_res_labels[high_res_labels == 5] = 4
        high_res_labels[high_res_labels == 6] = 4
        # reindex nodata value
        high_res_labels[high_res_labels == nodata_idx_pad_orig] = nodata_idx_pad_condensed
        high_res_labels = high_res_labels.long()


        sample["mask"] = prior
        # subtract 1 so it starts with 0
        sample["high_res_labels"] = high_res_labels - 1
        
        sample["image"] = sample["image"].float()

        #print(sample["image"].dtype, sample["image"].shape)
        #print(sample["mask"].dtype, sample["mask"].shape)
        #print(sample["high_res_labels"].dtype, sample["high_res_labels"].shape)
        del sample["crs"]
        del sample["bbox"]
        
        return dict(sample)
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.

        Args:
            stage: stage to set up
        """
        train_transforms = Compose(
            [
                self.center_crop(self.patch_size),
                self.nodata_check(self.patch_size),
                self.preprocess,
            ]
        )
        val_transforms = Compose(
            [
                self.center_crop(self.patch_size),
                self.nodata_check(self.patch_size),
                self.preprocess,
            ]
        )
        test_transforms = Compose(
            [
                # this is the only change from the torchgeo function -- mask value 7
                self.pad_to(self.original_patch_size, image_value=0, mask_value=7),
                self.preprocess,
            ]
        )

        self.train_dataset = ChesapeakeCVPR(
            self.root_dir,
            splits=self.train_splits,
            layers=self.layers,
            transforms=train_transforms,
            download=False,
            checksum=False,
        )
        self.val_dataset = ChesapeakeCVPR(
            self.root_dir,
            splits=self.val_splits,
            layers=self.layers,
            transforms=val_transforms,
            download=False,
            checksum=False,
        )
        self.test_dataset = ChesapeakeCVPR(
            self.root_dir,
            splits=self.test_splits,
            layers=self.layers,
            transforms=test_transforms,
            download=False,
            checksum=False,
        )
    
    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        sampler = GridGeoSampler(
            self.test_dataset,
            size=self.original_patch_size,
            stride=self.original_patch_size,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=16,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )