from turtle import forward
import torch
import torch.nn as nn
import tensorflow as tf

def padding1D(x, kernel_size, dilation=1):
    pad = ((kernel_size-1)*(2**dilation),0)
    out = torch.nn.functional.pad(x,pad)
    return out

class maxpool(nn.Module):
    def __init__(self):
        super(maxpool, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x): 
        x_sh = x.shape
        batch_size,electrode,in_channels,_ = x_sh
        
        x = x.reshape([batch_size*electrode, in_channels,-1])
        x = self.maxpool(x)
        x = x.reshape([batch_size,electrode, in_channels,-1])
        out = x.squeeze(dim=3)

        return out



class CausalBlock(nn.Module):
    def __init__(self, electrode, in_channels, mid_channels,out_channels, skip=False, kernel_size=3, dilation=1):
        super(CausalBlock, self).__init__()
        self.skip = skip
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.electrode = electrode

        if self.skip:
            self.skip_layer = nn.Conv1d(in_channels, out_channels, 1)
            self.layer1 = nn.Sequential(
                 #filter = K, kernel_size=3
                    torch.nn.utils.weight_norm(nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=2**dilation)),
                    #nn.BatchNorm1d(mid_channels),
                    nn.LeakyReLU(),
                
            )
            self.layer2 = nn.Sequential(
                    torch.nn.utils.weight_norm(nn.Conv1d(mid_channels, out_channels, kernel_size, dilation=2**dilation)),
                    #nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(),
                
            )

        else:
            self.layer1 = nn.Sequential(
                #filter = K, kernel_size=3
                    torch.nn.utils.weight_norm(nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=2**dilation)),
                    #nn.BatchNorm1d(mid_channels),
                    nn.LeakyReLU())
            self.layer2 =nn.Sequential(
                    torch.nn.utils.weight_norm(nn.Conv1d(mid_channels, out_channels, kernel_size, dilation=2**dilation)),
                    #nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(),
                
            )

    def forward(self, x):
        x_sh = x.shape
        batch_size = x_sh[0]
        x = x.reshape([batch_size*self.electrode, self.in_channels,-1])

        pad_x = padding1D(x, kernel_size=self.kernel_size, dilation=self.dilation)
        if self.skip:
            out = self.layer1(pad_x)
            out = padding1D(out, kernel_size=self.kernel_size, dilation=self.dilation)
            out = self.layer2(out)
            out += self.skip_layer(x)

            out = out.reshape([batch_size, self.electrode, self.out_channels,-1])
            return out
        else:
            out = self.layer1(pad_x)
            out = padding1D(out, kernel_size=self.kernel_size, dilation=self.dilation)
            out = self.layer2(out)

            out = out.reshape([batch_size, self.electrode, self.out_channels,-1])
            return out


class Encoder(nn.Module):
    def __init__(self, electrode, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.causalblock1 = CausalBlock(electrode, in_channels, 4, 4, skip=True, kernel_size=5, dilation=0)
        self.causalblock2 = CausalBlock(electrode, 4, 8, 8, skip=True, kernel_size=5, dilation=1)
        self.causalblock3 = CausalBlock(electrode, 8, 8, 16, skip=True, kernel_size=5, dilation=2)
        self.causalblock4 = CausalBlock(electrode, 16, 16, out_channels, skip=True, kernel_size=5, dilation=2)
        self.maxpool = maxpool()
        self.out_channels = out_channels
        self.electrode = electrode

    def forward(self, x):
        x = self.causalblock1(x)

        x1 = self.causalblock2(x)

        x2 = self.causalblock3(x1)
        f2 = self.maxpool(x1)

        x3 = self.causalblock4(x2)
        f3 = self.maxpool(x2)

        f4 = self.maxpool(x3)

        out = torch.cat([f2,f3,f4], -1)

        return out
