import torchvision
import torch.nn as nn
import torch

# 1x1 convolution
def conv1x1(in_channels, out_channels, stride, padding):
    model = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels)
    )
    return model


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride, padding):
    model = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels)
    )
    return model

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.layer = nn.Sequential(
                conv1x1( in_channels, middle_channels, 1, 0),
                conv3x3( middle_channels, middle_channels, 2, 1),
                conv1x1( middle_channels, out_channels, 1, 0)
            )
            self.downsize = conv1x1(in_channels, out_channels, 2, 0)

        else:
            self.layer = nn.Sequential(
                conv1x1( in_channels, middle_channels, 1, 0),
                conv3x3( middle_channels, middle_channels, 1, 1),
                conv1x1( middle_channels, out_channels, 1, 0)
            )
            self.make_equal_channel = conv1x1(in_channels, out_channels, 1, 0)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample:
            out = self.layer(x)
            x = self.downsize(x)
            return self.activation(out + x)
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.make_equal_channel(x)
            return self.activation(out + x)

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),  
        nn.Conv1d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )

class Resnet50Encoder(nn.Module):
    def __init__(self, inchannels = 12, n_classes=22):
        super().__init__()
        self.n_classes = n_classes
        self.layer1 = nn.Sequential(
            nn.Conv1d(inchannels, 16, kernel_size = 3, stride = 2, padding = 1 ), # Code overlaps with previous assignments
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool1d(3, 2, 1, return_indices=True)

        self.layer2 = nn.Sequential(
            ResidualBlock(16, 16, 32),
            ResidualBlock(32, 16, 32),
            ResidualBlock(32, 16, 32, downsample=True) # Code overlaps with previous assignments
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(32, 16, 64),
            ResidualBlock(64, 32, 64),
            ResidualBlock(64, 32, 64),
            ResidualBlock(64, 32, 64, downsample=False) # Code overlaps with previous assignments
        )
        self.bridge = conv(64, 64)
        self.fullconnect = nn.Linear(512,1)
    ###########################################################################
    # Question 2 : Implement the forward function of Resnet_encoder_UNet.
    # Understand ResNet, UNet architecture and fill in the blanks below.
    def forward(self, x): #256
        batch_size = x.shape[0]
        feature_size = x.shape[1]
        x = x.reshape([batch_size,1,feature_size])

        out1 = self.layer1(x)
        out1, indices = self.pool(out1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        x = self.bridge(out3) # bridge
        x = torch.flatten(x,1)
        x = self.fullconnect(x)
        x = torch.sigmoid(x)
        return x

