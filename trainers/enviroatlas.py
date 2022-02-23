# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for the Chesapeake datasets."""

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
from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
from torchmetrics import Accuracy, IoU
from torchvision.transforms import Compose

# todo remove local import once pip install works
import sys
sys.path.append('../scripts')
from qr_losses import loss_on_prior_reversed_kl_simple, loss_on_prior_simple
from fcn import FCN_modified

ENVIROATLAS_CLASS_COLORS_DICT = {
    0: (255, 255, 255, 255),  #
    1: (0, 197, 255, 255),  # from CC Water
    2: (156, 156, 156, 255),  # from CC Impervious
    3: (255, 170, 0, 255),  # from CC Barren
    4: (38, 115, 0, 255),  # from CC Tree Canopy
    5: (204, 184, 121, 255),  # from NLCD shrub
    6: (163, 255, 115, 255),  # from CC Low Vegetation
    7: (220, 217, 57, 255),  # from NLCD Pasture/Hay color
    8: (171, 108, 40, 255),  # from NLCD Cultivated Crops
    9: (184, 217, 235, 255),  # from NLCD Woody Wetlands
    10: (108, 159, 184, 255),  # from NLCD Emergent Herbaceous Wetlands
    11: (0, 0, 0, 0),  # extra for black
    12: (70, 100, 159, 255),  # extra for dark blue
}


def get_colors(class_colors):
    """Map colors dict to colors array."""
    return np.array([class_colors[c] for c in class_colors.keys()]) / 255.0


ENVIORATLAS_CLASS_COLORS = get_colors(ENVIROATLAS_CLASS_COLORS_DICT)


def vis_lc_from_colors(r, colors, renorm=True, reindexed=True):
    """Function for visualizing color scheme with potentially soft class assigments."""
    sparse = r.shape[0] != len(colors)
    colors_cycle = range(0, len(colors))

    if sparse:
        z = np.zeros((3,) + r.shape)
        s = r
        for c in colors_cycle:
            for ch in range(3):
                z[ch] += colors[c][ch] * (s == c).astype(float)

    else:
        z = np.zeros((3,) + r.shape[1:])
        if renorm:
            s = r / r.sum(0)
        else:
            s = r
        for c in colors_cycle:
            for ch in range(3):
                z[ch] += colors[c][ch] * s[c]
    return z


class EnviroatlasSegmentationTask(LightningModule):
    """LightningModule for training models on the Enviroatlas Land Cover dataset.

    This allows using arbitrary models and losses from the
    ``pytorch_segmentation_models`` package.
    """

    def config_task(self, kwargs: Dict[str, Any]) -> None:
        """Configures the task based on kwargs parameters."""
        self.classes_keep = kwargs["classes_keep"]
        self.colors = [ENVIORATLAS_CLASS_COLORS[c] for c in self.classes_keep]
        self.n_classes = len(self.classes_keep)
        self.n_classes_with_nodata = len(self.classes_keep) + 1
        self.ignore_index = len(self.classes_keep)

        if (
            "include_prior_as_datalayer" in kwargs.keys()
            and kwargs["include_prior_as_datalayer"]
        ):
            self.include_prior_as_datayer = True
            self.in_channels = 9  # 5 for prior, 4 for naip
        else:
            self.include_prior_as_datayer = False
            self.in_channels = 4
            
        # log the outputs if the loss is nll
        log_outputs = kwargs["loss"] == "nll"

        """Configures the task based on kwargs parameters."""
        if kwargs["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                activation=kwargs["activation_layer"],
                in_channels=self.in_channels,
                classes=self.n_classes,
            )
        elif kwargs["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                in_channels=self.in_channels,
                classes=self.n_classes,
            )
        elif kwargs["segmentation_model"] == "fcn":
            self.model = FCN_modified(
                in_channels=self.in_channels,
                classes=self.n_classes,
                num_filters=kwargs["num_filters"],
                output_smooth=kwargs["output_smooth"],
                log_outputs=log_outputs,
            )
            # 5 pixels per side
            self.pad = 5
        else:
            raise ValueError(
                f"Model type '{kwargs['segmentation_model']}' is not valid."
            )

        if kwargs["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss(  # type: ignore[attr-defined]
            )
        elif kwargs["loss"] == "nll":
            self.loss = nn.NLLLoss()

        elif kwargs["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass")
        else:
            raise ValueError(f"Loss type '{kwargs['loss']}' is not valid.")

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            loss: Name of the loss function
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task(kwargs)

        self.train_accuracy = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_accuracy = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_accuracy = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )

        self.train_iou = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_iou = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_iou = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_iou_per_class = IoU(
            num_classes=self.n_classes_with_nodata,
            ignore_index=self.ignore_index,
            reduction="none",
        )

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        return self.model(x)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"][:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat = self.forward(x)[:,:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False, batch_size=len(x))
        self.train_accuracy(y_hat_hard, y)
        self.train_iou(y_hat_hard, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics."""
        self.log("train_acc_q", self.train_accuracy.compute())
        self.log("train_iou_q", self.train_iou.compute())
        self.train_accuracy.reset()
        self.train_iou.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"][:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat = self.forward(x)[:,:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("val_loss", loss, batch_size=len(x))
        self.val_accuracy(y_hat_hard, y)
        self.val_iou(y_hat_hard, y)

        with torch.no_grad():
            if batch_idx < 10:
                # Render the image, ground truth mask, and predicted mask for the first
                # image in the batch
                img = np.rollaxis(  # convert image to channels last format
                    batch["image"][0][:4].cpu().numpy(), 0, 3
                )
                if self.include_prior_as_datayer:
                    prior = batch["image"][0][4:].cpu().numpy()
                    prior_vis = vis_lc_from_colors(prior, self.colors).T.swapaxes(0, 1)
                    img[:, :, :3] = (prior_vis + img[:, :, :3]) / 2.0

                # This is specific to the 5 class definition
                mask = batch["mask"][0,self.pad:-self.pad,self.pad:-self.pad].cpu().numpy()
                pred = y_hat_hard[0].cpu().numpy()
                if self.include_prior_as_datayer:
                    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
                    axs[0].imshow(img[:, :, :3])
                    axs[0].axis("off")
                    axs[1].imshow(prior_vis)
                    axs[1].axis("off")
                    axs[2].imshow(
                        vis_lc_from_colors(mask, self.colors).T.swapaxes(0, 1),
                        interpolation="none",
                    )
                    axs[2].axis("off")
                    axs[2].set_title("labels")
                    axs[3].imshow(
                        vis_lc_from_colors(pred, self.colors).T.swapaxes(0, 1),
                        interpolation="none",
                    )
                    axs[3].axis("off")
                    axs[3].set_title("predictions")
                    plt.tight_layout()
                else:
                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(img[:, :, :3])
                    axs[0].axis("off")
                    axs[1].imshow(
                        vis_lc_from_colors(mask, self.colors).T.swapaxes(0, 1),
                        interpolation="none",
                    )
                    axs[1].axis("off")
                    axs[1].set_title("labels")
                    axs[2].imshow(
                        vis_lc_from_colors(pred, self.colors).T.swapaxes(0, 1),
                        interpolation="none",
                    )
                    axs[2].axis("off")
                    axs[2].set_title("predictions")
                    plt.tight_layout()

                # the SummaryWriter is a tensorboard object, see:
                # https://pytorch.org/docs/stable/tensorboard.html#
                summary_writer: SummaryWriter = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics."""
        self.log("val_acc_q", self.val_accuracy.compute())
        self.log("val_iou_q", self.val_iou.compute())
        self.val_accuracy.reset()
        self.val_iou.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step."""
        x = batch["image"]
        y = batch["mask"][:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat = self.forward(x)[:,:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat_hard = y_hat.argmax(dim=1)

        #  deal with nodata in the test batches
        loss = 0

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss)
        self.test_accuracy(y_hat_hard, y)
        self.test_iou(y_hat_hard, y)
        self.test_iou_per_class(y_hat_hard, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics."""
        self.log("test_acc_q", self.test_accuracy.compute())
        self.log("test_iou_q", self.test_iou.compute())
        # print(self.test_iou_per_class.compute())
        self.log_dict(
            dict(
                zip(
                    [f"iou_{x}" for x in np.arange(self.n_classes)],
                    self.test_iou_per_class.compute(),
                )
            )
        )
        self.test_accuracy.reset()
        self.test_iou.reset()
        self.test_iou_per_class.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["learning_rate"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
                "verbose": True,
            },
        }


