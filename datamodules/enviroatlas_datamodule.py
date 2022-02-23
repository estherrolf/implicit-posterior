from typing import Any, Callable, Dict, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from torchgeo.datasets import EnviroAtlas
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers.single import GridGeoSampler
from torchgeo.samplers.batch import RandomBatchGeoSampler

class EnviroatlasDataModule(LightningDataModule):
    """LightningDataModule implementation for the Enviroatlas dataset.

    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """

    def __init__(
        self,
        root_dir: str,
        states_str: str,
        classes_keep: list,
        patches_per_tile: int = 200,
        patch_size: int = 128,
        batch_size: int = 64,
        num_workers: int = 4,
        train_set: str = "train",
        val_set: str = "val",
        test_set: str = "test",
        include_prior_as_datalayer=False,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Enviroatlas based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the Enviroatlas Dataset
            states_str: The states to use to train the model, concatenated with '+'
            patches_per_tile: The number of patches per tile to sample
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            patch_size: size of each instance in the batch, in pixels
            prior_version: string describing which prior to use
            classes_keep: list of valid classes for the prediction problem
            prior_smoothing_constant: additive smoothing constant
            train_set: Set to train on
            val_set:  Set to validate on
            test_set: Set to test on
            include_prior_as_datalayer: whether prior is input data or supervision data

        """
        super().__init__()  # type: ignore[no-untyped-call]

        states = states_str.split("+")
        for state in states:
            assert state in [
                "pittsburgh_pa-2010_1m",
                "durham_nc-2012_1m",
                "austin_tx-2012_1m",
                "phoenix_az-2010_1m",
            ]

        print(patches_per_tile)
        self.root_dir = root_dir
        self.include_prior_as_datalayer = include_prior_as_datalayer
        if self.include_prior_as_datalayer:
            self.layers = [
                "naip",
                "prior",
                "lc",
            ]
        else:
            self.layers = ["naip", "lc"]
        self.patches_per_tile = patches_per_tile
        self.patch_size = patch_size
        self.original_patch_size = patch_size * 3
        self.batch_size = batch_size
        self.num_workers = num_workers

        # if the prior is to be used, use it as input layer, not output supervision
        self.prior_as_input = True

        self.classes_keep = classes_keep
        self.ignore_index = len(classes_keep)

        self.train_sets = [f"{state}-{train_set}" for state in states]
        self.val_sets = [f"{state}-{val_set}" for state in states]
        self.test_sets = [f"{state}-{test_set}" for state in states]
        print(f"train sets are: {self.train_sets}")
        print(f"val sets are: {self.val_sets}")
        print(f"test sets are: {self.test_sets}")

    def pad_to(
        self, size: int = 512, image_value: int = 0, mask_value: int = 0
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a padding transform on a single sample.
        Args:
            size: output image size
            image_value: value to pad image with
            mask_value: value to pad mask with
        Returns:
            function to perform padding
        """

        def pad_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape
            assert height <= size and width <= size

            height_pad = size - height
            width_pad = size - width

            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            # for a description of the format of the padding tuple
            sample["image"] = F.pad(
                sample["image"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=image_value,
            )
            sample["mask"] = F.pad(
                sample["mask"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=mask_value,
            )
            return sample

        return pad_inner

    def center_crop(
        self, size: int = 512
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a center crop transform on a single sample.
        Args:
            size: output image size
        Returns:
            function to perform center crop
        """

        def center_crop_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape

            y1 = (height - size) // 2
            x1 = (width - size) // 2
            sample["image"] = sample["image"][:, y1 : y1 + size, x1 : x1 + size]
            sample["mask"] = sample["mask"][:, y1 : y1 + size, x1 : x1 + size]

            return sample

        return center_crop_inner

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesses a single sample."""
        # reindex to just the EnviroAtlas classes to be kept
        reindex_map = dict(zip(self.classes_keep, np.arange(len(self.classes_keep))))
        reindexed_mask = -1 * torch.ones(sample["mask"].shape)
        for old_idx, new_idx in reindex_map.items():
            reindexed_mask[sample["mask"] == old_idx] = new_idx

        reindexed_mask[reindexed_mask == -1] = self.ignore_index
        assert (reindexed_mask >= 0).all()

        sample["mask"] = reindexed_mask

        # note prior and images both should be divided by 255 as below
        sample["image"] = sample["image"] / 255.0
        # don't subtract 1 because we already reindexed
        sample["mask"] = sample["mask"].squeeze()

        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].long()
        
        del sample["bbox"]

        return sample
    
    def nodata_check(
        self, size: int = 512
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to check for nodata or mis-sized input.
        Args:
            size: output image size
        Returns:
            function to check for nodata values
        """

        def nodata_check_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            num_channels, height, width = sample["image"].shape

            if height < size or width < size:
                sample["image"] = torch.zeros(  # type: ignore[attr-defined]
                    (num_channels, size, size)
                )
                sample["mask"] = torch.zeros((size, size))  # type: ignore[attr-defined]

            return sample

        return nodata_check_inner

    def prepare_data(self) -> None:
        """Confirms that the dataset is downloaded on the local node.

        This method is called once per node, while :func:`setup` is called once per GPU.
        """
        EnviroAtlas(
            self.root_dir,
            splits=self.train_sets,
            layers=self.layers,
            prior_as_input=self.prior_as_input,
            transforms=None,
            download=False,
            checksum=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.
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
                self.pad_to(self.original_patch_size, image_value=0, mask_value=11),
                self.preprocess,
            ]
        )

        self.train_dataset = EnviroAtlas(
            self.root_dir,
            splits=self.train_sets,
            layers=self.layers,
            prior_as_input=self.prior_as_input,
            transforms=train_transforms,
            download=False,
            checksum=False,
        )
        self.val_dataset = EnviroAtlas(
            self.root_dir,
            splits=self.val_sets,
            layers=self.layers,
            prior_as_input=self.prior_as_input,
            transforms=val_transforms,
            download=False,
            checksum=False,
        )
        self.test_dataset = EnviroAtlas(
            self.root_dir,
            splits=self.test_sets,
            layers=self.layers,
            prior_as_input=self.prior_as_input,
            transforms=test_transforms,
            download=False,
            checksum=False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.
        Returns:
            training data loader
        """
        sampler = RandomBatchGeoSampler(
            self.train_dataset,
            size=self.original_patch_size,
            batch_size=self.batch_size,
            length=self.patches_per_tile * len(self.train_dataset),
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.
        Returns:
            validation data loader
        """
        sampler = GridGeoSampler(
            self.val_dataset,
            size=self.original_patch_size,
            stride=self.original_patch_size,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
            shuffle=False
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
            batch_size=64,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
            shuffle=False
        )
