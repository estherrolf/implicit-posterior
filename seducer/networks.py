import torch
import torch.nn.functional as F
import math

class FCN(torch.nn.Module):
    def __init__(self, input_ch, n_classes, conv_ch=16, pos_ch=4):
        '''Create FCN Q network.
        
        Args:
            input_ch (int): Number of input channels
            n_classes (int): Number of predicted classes
            conv_ch (int): Convolutional layer channels
            pos_ch (int): Position MLP channels
        '''
        super(FCN, self).__init__()
        # Network parameters
        self.input_ch = input_ch
        self.n_classes = n_classes
        self.conv_ch = conv_ch
        self.pos_ch = pos_ch

        # Image feature extraction
        self.conv1 = torch.nn.Conv2d(input_ch, conv_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(conv_ch, conv_ch,  3, 1, 1)
        self.conv3 = torch.nn.Conv2d(conv_ch, conv_ch,  3, 1, 1)
        self.conv4 = torch.nn.Conv2d(conv_ch, conv_ch,  3, 1, 1)
        
        self.norm1 = torch.nn.BatchNorm2d(conv_ch)
        self.norm2 = torch.nn.BatchNorm2d(conv_ch)
        self.norm3 = torch.nn.BatchNorm2d(conv_ch)
        self.norm4 = torch.nn.BatchNorm2d(conv_ch)
        
        # Positional encodings MLP
        self.pos_fourier = torch.nn.Conv2d(2, pos_ch, 1, bias=False)
        self.pos_mlp1 = torch.nn.Conv2d(2*pos_ch, pos_ch, 1)
        self.pos_mlp2 = torch.nn.Conv2d(pos_ch, pos_ch, 1)
        self.pos_mlp3 = torch.nn.Conv2d(pos_ch, pos_ch, 1)
        
        self.pos_norm1 = torch.nn.BatchNorm2d(pos_ch)
        self.pos_norm2 = torch.nn.BatchNorm2d(pos_ch)
        self.pos_norm3 = torch.nn.BatchNorm2d(pos_ch)
        
        # Output
        self.conv5 = torch.nn.Conv2d(conv_ch + pos_ch, conv_ch,  3, 1, 1)
        self.conv_out = torch.nn.Conv2d(conv_ch, n_classes, 1)
        
        self.norm5 = torch.nn.BatchNorm2d(conv_ch)
    
    def get_device(self):
        return self.conv1.weight.device
        
    def forward(self, img, pos_grid):
        '''Passes batch of image/position patches through network
        
        Args:
            img (torch.Tensor): Batch of image patches (batch_size,n_channels,height,width)
            pos_grid (torch.Tensor): Batch of position grid patches (batch_size,2,height,width)
        Returns:
            label_probs (torch.Tensor): Batch of predicted class probabilities (batch_size,n_classes,height,width)
        '''
        # Compute positional encoding features
        x = self.pos_fourier(pos_grid)
        x = torch.cat((torch.sin(x), torch.cos(x)), dim=1)
        x = F.relu(self.pos_norm1(self.pos_mlp1(x)))
        x = F.relu(self.pos_norm2(self.pos_mlp2(x)))
        pos_enc_features = F.relu(self.pos_norm3(self.pos_mlp3(x)))
        
        # Extract image features
        x = img
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        
        # Combine
        x = torch.cat((x, pos_enc_features), dim=1)
        x = F.relu(self.norm5(self.conv5(x)))
        label_scores = self.conv_out(x)
        label_probs = F.softmax(label_scores, dim=1)
        
        return label_probs
    