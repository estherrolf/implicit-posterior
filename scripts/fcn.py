# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Simple fully convolutional neural network (FCN) implementations."""

import torch as T
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"

class FCN_larger_modified(Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

    def __init__(self, 
                 in_channels: int, 
                 classes: int, 
                 num_filters: int = 128, 
                 output_smooth: float = 1e-2,
                 log_outputs: bool = True) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super(FCN_larger_modified, self).__init__()  # type: ignore[no-untyped-call]

        conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=11, stride=1, padding=5
        )
        conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=7, stride=1, padding=3
        )
        conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=5, stride=1, padding=2
        )

        self.backbone = nn.modules.Sequential(
            conv1,
            nn.modules.LeakyReLU(inplace=True),
            conv2,
            nn.modules.LeakyReLU(inplace=True),
            conv3
        )

        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )
        
        self.output_smooth = output_smooth
        self.log_outputs = log_outputs

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        # add smoothing
        x = self.last(x).softmax(1) + self.output_smooth
        # renormalize and log
        x = nn.functional.normalize(x, p=1, dim=1)
        if self.log_outputs:
            x = x.log()
        
        return x
    
    
class FCN_modified(Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

    def __init__(self, 
                 in_channels: int, 
                 classes: int, 
                 num_filters: int = 64, 
                 output_smooth: float = 1e-2,
                 log_outputs: bool = True) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super(FCN_modified, self).__init__()  # type: ignore[no-untyped-call]

        conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv4 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv5 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )

        self.backbone = nn.modules.Sequential(
            conv1,
            nn.modules.LeakyReLU(inplace=True),
            conv2,
            nn.modules.LeakyReLU(inplace=True),
            conv3,
            nn.modules.LeakyReLU(inplace=True),
            conv4,
            nn.modules.LeakyReLU(inplace=True),
            conv5,
            nn.modules.LeakyReLU(inplace=True),
        )

        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )
        
        self.output_smooth = output_smooth
        self.log_outputs = log_outputs

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        # add smoothing
        x = self.last(x).softmax(1) + self.output_smooth
        # renormalize and log
        x = nn.functional.normalize(x, p=1, dim=1)
        if self.log_outputs:
            x = x.log()
                
        return x

    
class FCN(Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

    def __init__(self, in_channels: int, classes: int, num_filters: int = 64) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super(FCN, self).__init__()

        conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv4 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv5 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )

        self.backbone = nn.modules.Sequential(
            conv1,
            nn.modules.LeakyReLU(inplace=True),
            conv2,
            nn.modules.LeakyReLU(inplace=True),
            conv3,
            nn.modules.LeakyReLU(inplace=True),
            conv4,
            nn.modules.LeakyReLU(inplace=True),
            conv5,
            nn.modules.LeakyReLU(inplace=True),
        )

        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        x = self.last(x)
        return x
    
class FCNBase_5layers_batchnorm_2(nn.Module):
    
    def __init__(self,
                 in_channels: int, 
                 classes: int, 
                 num_filters: int = 64, 
                 output_smooth: float = 1e-2,
                 log_outputs: bool = True) -> None:
        super(FCNBase_5layers_batchnorm_2,self).__init__()
        
        
        # max pooling adds smoothing
        # some with max pooling, some without max pooling
        self.num_outputs = classes
        self.in_channels = in_channels
        self.nc = num_filters
        self.output_smooth = output_smooth
                
        self.conv1 = nn.Conv2d(self.in_channels,self.nc,3,1,1,1)
        self.conv2 = nn.Conv2d(self.nc,self.nc,3,1,1,1)
        self.conv3 = nn.Conv2d(self.nc,self.nc,3,1,1,1)
        self.conv4 = nn.Conv2d(self.nc,self.nc,3,1,1,1)
        self.conv5 = nn.Conv2d(self.nc,self.nc,3,1,1,1)
        
        self.norm1 = nn.BatchNorm2d(self.nc)
        self.norm2 = nn.BatchNorm2d(self.nc)
        self.norm3 = nn.BatchNorm2d(self.nc)
        self.norm4 = nn.BatchNorm2d(self.nc)
        self.norm5 = nn.BatchNorm2d(self.nc)
        
        self.lastweight = nn.Parameter(T.zeros(self.nc,self.num_outputs).uniform_())
        self.lastbias = nn.Parameter(T.zeros(self.num_outputs).uniform_())
        
        self.log_outputs = log_outputs
        
    def forward(self,inputs):
        x = T.relu(self.norm1(self.conv1(inputs)))
        x = T.relu(self.norm2(self.conv2(x)))
        x = T.relu(self.norm3(self.conv3(x)))
        x = T.relu(self.norm4(self.conv4(x)))
        x = T.relu(self.norm5(self.conv5(x)))

        outs = (T.einsum('bfxy,fo->boxy',x,self.lastweight) + \
                self.lastbias.unsqueeze(1).unsqueeze(2)).softmax(1) + self.output_smooth

        # make sure the outputs are log probabilities
        outs = outs / outs.sum(1).unsqueeze(1)
        print(outs.shape)
        print(self.log_outputs)
        if self.log_outputs:
    #        print(outs.log())
            return outs.log()
        else:
            return outs

class FCN_modified_test_batchnorm(Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

    def __init__(self, 
                 in_channels: int, 
                 classes: int, 
                 num_filters: int = 64, 
                 output_smooth: float = 1e-2,
                 log_outputs: bool = True) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super(FCN_modified_test_batchnorm, self).__init__()  # type: ignore[no-untyped-call]

        conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv4 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv5 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        
        norm1 = nn.BatchNorm2d(num_filters)
        norm2 = nn.BatchNorm2d(num_filters)
        norm3 = nn.BatchNorm2d(num_filters)
        norm4 = nn.BatchNorm2d(num_filters)
        norm5 = nn.BatchNorm2d(num_filters)
        

        self.backbone = nn.modules.Sequential(
            conv1,
            norm1,
            nn.modules.LeakyReLU(inplace=True),
            conv2,
            norm2,
            nn.modules.LeakyReLU(inplace=True),
            conv3,
            norm3,
            nn.modules.LeakyReLU(inplace=True),
            conv4,
            norm4,
            nn.modules.LeakyReLU(inplace=True),
            conv5,
            norm5,
            nn.modules.LeakyReLU(inplace=True),
        )

        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )
        
        self.output_smooth = output_smooth
        self.log_outputs = log_outputs

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        # add smoothing
        x = self.last(x).softmax(1) + self.output_smooth
        # renormalize and log
        x = nn.functional.normalize(x, p=1, dim=1)
        if self.log_outputs:
            x = x.log()
                
        return x
    
