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

import sys
sys.path.append('../scripts')
from fcn import FCN_larger_modified, FCN_modified

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"

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


ENVIROATLAS_CLASS_COLORS = get_colors(ENVIROATLAS_CLASS_COLORS_DICT)


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


class EnviroatlasLearnPriorTask(LightningModule):
    """LightningModule for training models on the Enviroatlas dataset.

    This allows using arbitrary models and losses from the
    ``pytorch_segmentation_models`` package.
    """

    def config_task(self, kwargs: Dict[str, Any]) -> None:
        """Configures the task based on kwargs parameters."""
        self.classes_keep = kwargs["classes_keep"]
        self.colors = [ENVIROATLAS_CLASS_COLORS[c] for c in self.classes_keep]
        self.n_classes = len(self.classes_keep)
        self.n_classes_with_nodata = len(self.classes_keep) + 1
        self.ignore_index = len(self.classes_keep)

        self.in_channels = 9  # 5 for prior, 4 for naip

        # five from the blurred NLCD remapped to EA5, plus
        # roads, buildings, waterways and waterbodies
        self.in_channels = 9
        
        self.labels_are_onehot = False
        self.need_to_pad_output_with_zeros = False
        
        # log the outputs if the loss is nll
        log_outputs = kwargs["loss"] == "nll"
        self.need_to_exp_outputs = log_outputs
            

        """Configures the task based on kwargs parameters."""
        if kwargs["segmentation_model"] == "fcn":
            self.model = FCN_modified(
                in_channels=self.in_channels,
                classes=self.n_classes,
                num_filters=kwargs["num_filters"],
                output_smooth=kwargs["output_smooth"],
                log_outputs=log_outputs,
            )
            self.pad = 5
        elif kwargs["segmentation_model"] == "fcn_larger":
            self.model = FCN_larger_modified(
                in_channels=self.in_channels,
                classes=self.n_classes,
                num_filters=kwargs["num_filters"],
                output_smooth=kwargs["output_smooth"],
                log_outputs=log_outputs,
            )
            self.pad = 10
        else:
            raise ValueError(
                f"Model type '{kwargs['segmentation_model']}' is not valid."
            )

        if kwargs["loss"] == "nll":
            self.loss = nn.NLLLoss(ignore_index=self.ignore_index)
            self.need_to_pad_output_with_zeros = True

        elif kwargs["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss()

        elif kwargs["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass")

        elif kwargs["loss"] == "l2":
            self.loss = nn.MSELoss(reduction="mean")
            self.labels_are_onehot = True

        else:
            raise ValueError(f"Loss type '{kwargs['loss']}' is not valid.")

        print(self.loss)

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

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        if self.need_to_pad_output_with_zeros:
            model_out = self.model(x)
            out_shape = list(model_out.shape)
            out_shape[1] = 1
            # add zeros in case there are nodata in the labels
            zeros_to_match = torch.zeros(out_shape).to(model_out.get_device())
            return torch.cat((model_out, zeros_to_match), dim=1)
        else:
            return self.model(x)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"][:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat = self.forward(x)[:,:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat_hard = y_hat[:, : self.n_classes].argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False, batch_size=len(x))
        if self.labels_are_onehot:
            self.train_accuracy(y_hat_hard, y.argmax(dim=1))
            self.train_iou(y_hat_hard, y.argmax(dim=1))
        else:
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
        y_hat_hard = y_hat[:, : self.n_classes].argmax(dim=1)

        #    print(y_hat.shape)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("val_loss", loss, batch_size=len(x))
        if self.labels_are_onehot:
            self.val_accuracy(y_hat_hard, y.argmax(dim=1))
            self.val_iou(y_hat_hard, y.argmax(dim=1))
        else:
            self.val_accuracy(y_hat_hard, y)
            self.val_iou(y_hat_hard, y)

        if batch_idx < 10:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            inputs = batch["image"][0].cpu().numpy()
            mask = batch["mask"][0,self.pad:-self.pad,self.pad:-self.pad].cpu().numpy()
            
            if self.need_to_exp_outputs:
                q = torch.exp(y_hat[0][: self.n_classes]).cpu().numpy()
            else:
                q = (y_hat[0][: self.n_classes]).cpu().numpy()
            # squish the input layers of the image according to the assumption
            # that they are in this order:
            # [f"nlcd_onehot_blurred_kernelsize_{nlcd_blur_kernelsize}_sigma_{nlcd_blur_sigma}",
            #           "buildings", "roads", "waterbodies", "waterways", "lc"]
            squished_layers = inputs.copy()
            squished_layers[1] += inputs[5:7].sum(axis=0)  # buildings and roads
            squished_layers[0] += inputs[7:9].sum(axis=0)  # water

            input_vis = vis_lc_from_colors(
                squished_layers[:5] / squished_layers[:5].sum(axis=0), self.colors
            ).T.swapaxes(0, 1)
            pred_vis = vis_lc_from_colors(q, self.colors).T.swapaxes(0, 1)
            label_vis = vis_lc_from_colors(mask, self.colors).T.swapaxes(0, 1)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(input_vis, interpolation="none")
            axs[0].axis("off")
            axs[1].imshow(pred_vis, interpolation="none")
            axs[1].axis("off")
            axs[2].imshow(label_vis, interpolation="none")
            axs[2].axis("off")
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
        y_hat_hard = y_hat[:, : self.n_classes].argmax(dim=1)
        loss = 0
        # loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, batch_size=len(x))

        if self.labels_are_onehot:
            self.test_accuracy(y_hat_hard, y.argmax(dim=1))
            self.test_iou(y_hat_hard, y.argmax(dim=1))
        else:
            self.test_accuracy(y_hat_hard, y)
            self.test_iou(y_hat_hard, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics."""
        self.log("test_acc_q", self.test_accuracy.compute())
        self.log("test_iou_q", self.test_iou.compute())
        self.test_accuracy.reset()
        self.test_iou.reset()

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



