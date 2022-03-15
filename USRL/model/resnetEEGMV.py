from pickle import TRUE
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
            return self.activation(out)
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.make_equal_channel(x)
            return self.activation(out)

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
        #out_channel 32
        self.layer1 = nn.Sequential(
            nn.Conv1d(inchannels, 4, kernel_size = 5, stride = 1, padding = 2 ), # Code overlaps with previous assignments
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool1d(3, 2, 1, return_indices=True)
        '''
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, 64),
            ResidualBlock(64, 64, 64),
            ResidualBlock(64, 64, 96, downsample=True) # Code overlaps with previous assignments
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(96, 96, 96),
            ResidualBlock(96, 96, 96),
            ResidualBlock(96, 96, 96, downsample=True) # Code overlaps with previous assignments
        )'''
        
        self.layer2 = nn.Sequential(
            ResidualBlock(4, 8, 8),
            ResidualBlock(8, 8, 8, downsample=True),
            ResidualBlock(8, 16, 16),
            ResidualBlock(16, 16, 16, downsample=True) # Code overlaps with previous assignments
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(16, 32, 32),
            ResidualBlock(32, 32, 64, downsample=True),
            ResidualBlock(64, 128, 128),
            ResidualBlock(128, 128, 128, downsample=True) # Code overlaps with previous assignments
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(128, 128, 128),
            ResidualBlock(128, 128, 256, downsample=True),
            ResidualBlock(256, 256, 256),
            ResidualBlock(256, 256, 512, downsample=True) # Code overlaps with previous assignments
        )
        #out channel 64
        '''
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, 64),
            ResidualBlock(64, 88, 88, downsample=True),
            ResidualBlock(88, 88, 88),
            ResidualBlock(88, 88, 96, downsample=True) # Code overlaps with previous assignments
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(96, 96, 96),
            ResidualBlock(96, 108, 108, downsample=True),
            ResidualBlock(108, 108, 108),
            ResidualBlock(108, 108, 128, downsample=True) # Code overlaps with previous assignments
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(128, 128, 128),
            ResidualBlock(128, 144, 144, downsample=True),
            ResidualBlock(144, 144, 144),
            ResidualBlock(144, 156, 156, downsample=True) # Code overlaps with previous assignments
        )
        '''
        
        # out_channel 32
        '''
        self.layer1 = nn.Sequential(
            nn.Conv1d(inchannels, 32, kernel_size = 3, stride = 2, padding = 1 ), # Code overlaps with previous assignments
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool1d(3, 2, 1, return_indices=True)

        self.layer2 = nn.Sequential(
            ResidualBlock(32, 32, 48, downsample=True),
            ResidualBlock(48, 48, 64, downsample=True),
            ResidualBlock(64, 64, 80, downsample=True) # Code overlaps with previous assignments
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(80, 80, 96, downsample=True),
            ResidualBlock(96, 96, 108, downsample=True),
            ResidualBlock(108, 108, 128, downsample=True) # Code overlaps with previous assignments
        )'''

        self.bridge = conv(512, 512)
        self.fullconnect = nn.Sequential(
                    torch.nn.utils.weight_norm(nn.Linear(1024, 1024)),
                    nn.LeakyReLU(),
                    #nn.Linear(128, 128),
                    #nn.LeakyReLU(),
                    torch.nn.utils.weight_norm(nn.Linear(1024, 1024)),
                    nn.LeakyReLU(),
                    torch.nn.utils.weight_norm(nn.Linear(1024, 1024)),
                    nn.LeakyReLU(),
                    torch.nn.utils.weight_norm(nn.Linear(1024, 5)),
                    nn.LeakyReLU(),
                )
        # Dropout
        self.dropout = nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)
    ###########################################################################
    # Question 2 : Implement the forward function of Resnet_encoder_UNet.
    # Understand ResNet, UNet architecture and fill in the blanks below.
    def forward(self, x): #256
        
        x = x.unsqueeze(dim=1)
        x = self.layer1(x)
        x, indices = self.pool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bridge(x) # bridge
        x = torch.flatten(x,1)
        x = self.fullconnect(x)
        return x

