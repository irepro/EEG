import torch
import torch.nn as nn
import tensorflow as tf

def padding1D(x, electrode, in_channels, kernel_size, dilation=1):
    pad = ((kernel_size-1)*(2**(dilation+1)),0)
    out = torch.nn.functional.pad(x,pad)
    out = torch.reshape(out, [electrode, in_channels, -1])
    return out

class CausalBlock(nn.Module):
    def __init__(self, electrode, in_channels, mid_channels,out_channels, skip=False, kernel_size=3, dilation=1):
        super(CausalBlock, self).__init__()
        self.skip = skip
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.electrode = electrode

        if self.skip:
            self.skip_layer = nn.Conv1d(in_channels, out_channels, 1)
            self.layer = nn.Sequential(
                 #filter = K, kernel_size=3
                    torch.nn.utils.weight_norm(nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=2**dilation)),
                    #nn.BatchNorm1d(mid_channels),
                    nn.LeakyReLU(),
                    torch.nn.utils.weight_norm(nn.Conv1d(mid_channels, out_channels, kernel_size, dilation=2**dilation)),
                    #nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(),
                
            )

        else:
            self.layer = nn.Sequential(
                #filter = K, kernel_size=3
                    torch.nn.utils.weight_norm(nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=2**dilation)),
                    #nn.BatchNorm1d(mid_channels),
                    nn.LeakyReLU(),
                    torch.nn.utils.weight_norm(nn.Conv1d(mid_channels, out_channels, kernel_size, dilation=2**dilation)),
                    #nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(),
                
            )

    def forward(self, x):
        pad_x = padding1D(x, electrode = self.electrode, in_channels=self.in_channels, kernel_size=self.kernel_size, dilation=self.dilation)
        if self.skip:
            out = self.layer(pad_x)
            x = torch.reshape(x, [self.electrode, self.in_channels,-1])
            x = self.skip_layer(x)
            return out + x
        else:
            out = self.layer(pad_x)
            return out


class Encoder(nn.Module):
    def __init__(self, electrode, in_channels, out_channels):
        super(Encoder, self).__init__()
        #self.causalblock1 = CausalBlock(in_channels, 10, 10, skip=True, kernel_size=3, dilation=0)
        #self.causalblock2 = CausalBlock(10, 10, 10, skip=True, kernel_size=3, dilation=1)
        #self.causalblock3 = CausalBlock(10, 10, out_channels, skip=True, kernel_size=3, dilation=2)
        #self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.causalblock1 = CausalBlock(electrode, in_channels, 4, 8, skip=True, kernel_size=3, dilation=0)
        self.causalblock2 = CausalBlock(electrode, 8, 16, 32, skip=True, kernel_size=3, dilation=1)
        self.causalblock3 = CausalBlock(electrode, 32, 64, out_channels, skip=True, kernel_size=3, dilation=2)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.out_channels = out_channels
        self.electrode = electrode

    def forward(self, x):
        x1 = self.causalblock1(x)
        x2 = self.causalblock2(x1)
        x3 = self.causalblock3(x2)
        out = self.maxpool(x3)
        out = out.squeeze(dim=2)

        return out
