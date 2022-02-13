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
from qr_losses import loss_on_prior_reversed_kl_simple, loss_on_prior_simple
from fcn import FCN_modified, FCN_modified_batchnorm

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


class EnviroatlasPriorSegmentationTask(LightningModule):
    """LightningModule for training models on the EnviroAtlas Land Cover dataset.

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

        self.in_channels = 4

        qr_losses = ["qr_forward", "qr_reverse"]
        self.need_to_add_smoothing = (kwargs["segmentation_model"] != "fcn") and (
            kwargs["loss"] in qr_losses
        )
        if self.need_to_add_smoothing:
            print("will add smoothing after softmax")
            self.output_smooth = kwargs["output_smooth"]

        print(self.n_classes)
        if kwargs["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                in_channels=4,
                classes=self.n_classes,
                activation="softmax",
            )
        elif kwargs["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                in_channels=4,
                classes=self.n_classes,
            )
        elif kwargs["segmentation_model"] == "fcn":
            self.model = FCN_modified(
                in_channels=4,
                classes=self.n_classes,
                num_filters=kwargs["num_filters"],
                output_smooth=kwargs["output_smooth"],
            )
            self.pad = 5
        else:
            raise ValueError(
                f"Model type '{kwargs['segmentation_model']}' is not valid."
            )

        if kwargs["loss"] == "qr_forward":
            # self.loss = loss_on_prior_simple
            self.loss = loss_on_prior_simple
        elif kwargs["loss"] == "qr_reverse":
            self.loss = loss_on_prior_reversed_kl_simple
        elif kwargs["loss"] == "nll":
            self.loss = nn.NLLLoss()
        else:
            raise ValueError(f"Loss type '{kwargs['loss']}' is not valid.")

    def update_model_from_checkpoint(self, model_ckpt):
        """Initializes with model from a checkpoint."""
        print(f"using checkpoint from: {model_ckpt}")
        state_dict = torch.load(model_ckpt)["state_dict"]

        new_dict = {}
        for key, value in state_dict.items():
            new_dict[key.replace("model.", "")] = value

        self.model.load_state_dict(new_dict)

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
        print(kwargs)
        self.config_task(kwargs)

        initialize_from_checkpoint = "model_ckpt" in kwargs.keys()
        if initialize_from_checkpoint:
            self.update_model_from_checkpoint(kwargs["model_ckpt"])

        self.train_accuracy_q = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_accuracy_q = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_accuracy_q = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )

        self.train_iou_q = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_iou_q = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_iou_q = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )

        self.train_accuracy_r = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_accuracy_r = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_accuracy_r = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )

        self.train_iou_r = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_iou_r = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_iou_r = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_iou_q_per_class = IoU(
            num_classes=self.n_classes_with_nodata,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        self.test_iou_r_per_class = IoU(
            num_classes=self.n_classes_with_nodata,
            ignore_index=self.ignore_index,
            reduction="none",
        )

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        preds = self.model(x)

        if self.need_to_add_smoothing:
            preds = nn.functional.normalize(
                preds + self.output_smooth, p=1, dim=1
            ).log()
        return preds

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"][:,:,self.pad:-self.pad,self.pad:-self.pad]
        y_hr = batch["high_res_labels"][:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat = self.forward(x)[:,:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        with torch.no_grad():
            z = nn.functional.normalize(torch.exp(y_hat), p=1, dim=(0, 2, 3))
            # y is the prior
            r_hat_hard = (z * y).argmax(dim=1)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        self.train_accuracy_q(y_hat_hard, y_hr)
        self.train_iou_q(y_hat_hard, y_hr)
        self.train_accuracy_r(r_hat_hard, y_hr)
        self.train_iou_r(r_hat_hard, y_hr)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics."""
        self.log("train_acc_q", self.train_accuracy_q.compute())
        self.log("train_acc_r", self.train_accuracy_r.compute())
        self.log("train_iou_q", self.train_iou_q.compute())
        self.log("train_iou_r", self.train_iou_r.compute())
        self.train_accuracy_q.reset()
        self.train_accuracy_r.reset()
        self.train_iou_q.reset()
        self.train_iou_r.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"][:,:,self.pad:-self.pad,self.pad:-self.pad]
        y_hr = batch["high_res_labels"][:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat = self.forward(x)[:,:,self.pad:-self.pad,self.pad:-self.pad]
        
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        with torch.no_grad():
            z = nn.functional.normalize(torch.exp(y_hat), p=1, dim=(0, 2, 3))
            r_hat_hard = (z * y).argmax(dim=1)

        # by default, the test and validation steps only log per *epoch*
        self.log("val_loss", loss)
        self.val_accuracy_q(y_hat_hard, y_hr)
        self.val_iou_q(y_hat_hard, y_hr)

        self.val_accuracy_r(r_hat_hard, y_hr)
        self.val_iou_r(r_hat_hard, y_hr)

        if batch_idx < 10:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            img = np.rollaxis(  # convert image to channels last format
                batch["image"][0].cpu().numpy(), 0, 3
            )

            prior = batch["mask"][0,:,self.pad:-self.pad,self.pad:-self.pad]

            prior_vis = vis_lc_from_colors(prior.cpu().numpy(), self.colors).T.swapaxes(
                0, 1
            )
            high_res_labels_vis = vis_lc_from_colors(
                batch["high_res_labels"][0].cpu().numpy(), self.colors
            ).T.swapaxes(0, 1)

            q = torch.exp(y_hat[0])
            pred_vis = vis_lc_from_colors(q.cpu().numpy(), self.colors).T.swapaxes(0, 1)
            # calculated r (one one image, so classes are on dim 0)
            r = nn.functional.normalize(z[0] * prior, p=1, dim=0)
            r_vis = vis_lc_from_colors(r.cpu().numpy(), self.colors).T.swapaxes(0, 1)

            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            axs[0].imshow(img[:, :, :3])
            axs[0].set_title("NAIP")
            axs[0].axis("off")
            axs[1].imshow(prior_vis, interpolation="none")
            axs[1].set_title("prior")
            axs[1].axis("off")
            axs[2].imshow(pred_vis, interpolation="none")
            axs[2].set_title("q()")
            axs[2].axis("off")
            plt.tight_layout()
            axs[3].imshow(r_vis, interpolation="none")
            axs[3].set_title("r = z(q)*prior")
            axs[3].axis("off")
            axs[4].set_title("high res labels (CS)")
            axs[4].imshow(high_res_labels_vis, interpolation="none")
            axs[4].axis("off")

            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics."""
        self.log("val_acc_q", self.val_accuracy_q.compute())
        self.log("val_iou_q", self.val_iou_q.compute())
        self.log("val_acc_r", self.val_accuracy_r.compute())
        self.log("val_iou_r", self.val_iou_r.compute())
        self.val_accuracy_q.reset()
        self.val_accuracy_r.reset()
        self.val_iou_q.reset()
        self.val_iou_r.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step."""
        x = batch["image"]
        y = batch["mask"][:,:,self.pad:-self.pad,self.pad:-self.pad]
        y_hr = batch["high_res_labels"][:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat = self.forward(x)[:,:,self.pad:-self.pad,self.pad:-self.pad]
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        with torch.no_grad():
            z = nn.functional.normalize(torch.exp(y_hat), p=1, dim=(0, 2, 3))
            r_hat_hard = (z * y).argmax(dim=1)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss)
        self.test_accuracy_q(y_hat_hard, y_hr)
        self.test_iou_q(y_hat_hard, y_hr)
        self.test_accuracy_r(r_hat_hard, y_hr)
        self.test_iou_r(r_hat_hard, y_hr)
        self.test_iou_q_per_class(y_hat_hard, y_hr)
        self.test_iou_r_per_class(r_hat_hard, y_hr)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics."""
        self.log("test_acc_q", self.test_accuracy_q.compute())
        self.log("test_acc_r", self.test_accuracy_r.compute())
        self.log("test_iou_q", self.test_iou_q.compute())
        self.log("test_iou_r", self.test_iou_r.compute())
        self.log_dict(
            dict(
                zip(
                    [f"iou_q_{x}" for x in np.arange(self.n_classes)],
                    self.test_iou_q_per_class.compute(),
                )
            )
        )
        self.log_dict(
            dict(
                zip(
                    [f"iou_r_{x}" for x in np.arange(self.n_classes)],
                    self.test_iou_r_per_class.compute(),
                )
            )
        )
        self.test_accuracy_q.reset()
        self.test_accuracy_r.reset()
        self.test_iou_q.reset()
        self.test_iou_r.reset()
        self.test_iou_q_per_class.reset()
        self.test_iou_r_per_class.reset()

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

