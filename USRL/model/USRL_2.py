from queue import Full
from turtle import forward
import torch
import torch.nn as nn
from model import resnetEEGMV, resnetEEG


def padding1D(x, kernel_size, dilation=1):
    pad = ((kernel_size - 1) * (2 ** (dilation + 1)), 0)
    out = torch.nn.functional.pad(x, pad)
    return out


class maxpool(nn.Module):
    def __init__(self):
        super(maxpool, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x_sh = x.shape
        batch_size, electrode, in_channels, _ = x_sh

        x = x.reshape([batch_size * electrode, in_channels, -1])
        x = self.maxpool(x)
        x = x.reshape([batch_size, electrode, in_channels])
        # x = x.squeeze(dim=3)

        return x


class maxpool_Full_elec(nn.Module):
    def __init__(self):
        super(maxpool_Full_elec, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x_sh = x.shape
        batch_size, in_channels, _ = x_sh

        x = x.reshape([batch_size, in_channels, -1])
        x = self.maxpool(x)
        x = x.reshape([batch_size, in_channels, -1])
        x = x.squeeze(dim=2)

        return x


class causalMaxPool(nn.Module):
    def __init__(self):
        super(causalMaxPool, self).__init__()
        self.maxpool = nn.MaxPool1d(2)

    def forward(self, x, Full_elec=False):
        if Full_elec:
            x = self.maxpool(x)
        else:
            x = torch.squeeze(x, dim=0)
            x = self.maxpool(x)
            x = torch.unsqueeze(x, dim=0)

        return x


class CausalBlock(nn.Module):
    def __init__(self, electrode, in_channels, mid_channels, out_channels, skip=False, kernel_size=3, dilation=1):
        super(CausalBlock, self).__init__()
        self.skip = skip
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.electrode = electrode

        if self.skip:
            self.skip_layer = nn.Conv1d(in_channels, out_channels, 1)
            self.layer = nn.Sequential(
                # filter = K, kernel_size=3
                torch.nn.utils.weight_norm(nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=2 ** dilation)),
                # nn.BatchNorm1d(mid_channels),
                nn.LeakyReLU(),
                torch.nn.utils.weight_norm(nn.Conv1d(mid_channels, out_channels, kernel_size, dilation=2 ** dilation)),
                # nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(),

            )

        else:
            self.layer = nn.Sequential(
                # filter = K, kernel_size=3
                torch.nn.utils.weight_norm(nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=2 ** dilation)),
                # nn.BatchNorm1d(mid_channels),
                nn.LeakyReLU(),
                torch.nn.utils.weight_norm(nn.Conv1d(mid_channels, out_channels, kernel_size, dilation=2 ** dilation)),
                # nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(),

            )

    def forward(self, x):
        x_sh = x.shape
        batch_size = x_sh[0]
        x = x.reshape([batch_size * self.electrode, self.in_channels, -1])

        pad_x = padding1D(x, kernel_size=self.kernel_size, dilation=self.dilation)
        if self.skip:
            out = self.layer(pad_x)
            out += self.skip_layer(x)
            out = out.reshape([batch_size, self.electrode, self.out_channels, -1])
            return out
        else:
            out = self.layer(pad_x)
            out = out.reshape([batch_size, self.electrode, self.out_channels, -1])
            return out


class CausalBlock_MV(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, skip=False, kernel_size=3, dilation=1):
        super(CausalBlock_MV, self).__init__()
        self.skip = skip
        self.dilation = dilation
        self.kernel_size = kernel_size
        # self.electrode = electrode
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.skip:
            self.skip_layer = nn.Conv1d(in_channels, out_channels, 1)
            self.layer = nn.Sequential(
                # filter = K, kernel_size=3
                torch.nn.utils.weight_norm(nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=2 ** dilation)),
                # nn.BatchNorm1d(mid_channels),
                nn.LeakyReLU(),
                torch.nn.utils.weight_norm(nn.Conv1d(mid_channels, out_channels, kernel_size, dilation=2 ** dilation)),
                # nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(),

            )

        else:
            self.layer = nn.Sequential(
                # filter = K, kernel_size=3
                torch.nn.utils.weight_norm(nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=2 ** dilation)),
                # nn.BatchNorm1d(mid_channels),
                nn.LeakyReLU(),
                torch.nn.utils.weight_norm(nn.Conv1d(mid_channels, out_channels, kernel_size, dilation=2 ** dilation)),
                # nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(),

            )

    def forward(self, x):
        x_sh = x.shape
        batch_size = x_sh[0]
        x = x.reshape([batch_size, self.in_channels, -1])

        pad_x = padding1D(x, kernel_size=self.kernel_size, dilation=self.dilation)
        if self.skip:
            out = self.layer(pad_x)
            out += self.skip_layer(x)
            out = out.reshape([batch_size, self.out_channels, -1])
            return out
        else:
            out = self.layer(pad_x)
            out = out.reshape([batch_size, self.out_channels, -1])
            return out


class Encoder(nn.Module):
    def __init__(self, electrode, in_channels, out_channels, Full_elec):
        super(Encoder, self).__init__()
        if Full_elec:
            self.causalblock1 = CausalBlock_MV(electrode, 64, 64, skip=True, kernel_size=5, dilation=0)
            self.causalblock2 = CausalBlock_MV(64, 96, 96, skip=True, kernel_size=5, dilation=1)
            self.causalblock3 = CausalBlock_MV(96, 128, 128, skip=True, kernel_size=5, dilation=2)
            self.causalblock4 = CausalBlock_MV(128, 196, 196, skip=True, kernel_size=5, dilation=3)
            self.causalblock5 = CausalBlock_MV(196, out_channels, out_channels, skip=True, kernel_size=5, dilation=4)
            # self.causalblock5 = CausalBlock_MV(512, out_channels, out_channels, skip=True, kernel_size=5, dilation=3)

            self.maxpool = maxpool_Full_elec()
        else:
            self.causalblock1 = CausalBlock_MV(electrode, in_channels, 4, 4, skip=True, kernel_size=5, dilation=0)
            self.causalblock2 = CausalBlock_MV(electrode, 4, 8, 8, skip=True, kernel_size=5, dilation=1)
            self.causalblock3 = CausalBlock_MV(electrode, 8, 16, 16, skip=True, kernel_size=5, dilation=2)
            self.causalblock4 = CausalBlock_MV(electrode, 16, 32, 32, skip=True, kernel_size=5, dilation=3)
            self.causalblock5 = CausalBlock_MV(electrode, 32, 64, 64, skip=True, kernel_size=5, dilation=4)
            self.causalblock6 = CausalBlock_MV(electrode, 64, out_channels, out_channels, skip=True, kernel_size=5,
                                               dilation=6)
            self.maxpool = maxpool()
        self.out_channels = out_channels
        self.maxpool1d = causalMaxPool()
        self.electrode = electrode
        self.Full_elec = Full_elec

    def forward(self, x):
        if self.Full_elec:
            x = x.unsqueeze(dim=1)
        x = self.causalblock1(x)
        x = self.maxpool1d(x, self.Full_elec)

        x = self.causalblock2(x)
        f1 = self.maxpool(x)
        x = self.maxpool1d(x, self.Full_elec)

        x = self.causalblock3(x)
        f2 = self.maxpool(x)
        x = self.maxpool1d(x, self.Full_elec)

        x = self.causalblock4(x)
        f3 = self.maxpool(x)
        x = self.maxpool1d(x, self.Full_elec)

        x = self.causalblock5(x)
        f4 = self.maxpool(x)
        # x = self.causalblock6(x)

        #out = self.maxpool(x)

        out = torch.cat([f1, f2, f3, f4], -1)

        return out


class USRL(nn.Module):
    ####################################################
    # electorde = the number of electrode
    # in_channels, out_channels = the number of channel
    # Full_elec = True or False about whether you use all electrode data in encoding representation
    # During Unsupervised learning, parameter 'Unsupervise' is True. Then, change parameter to False when you start to supervised learning
    # ###################################################
    def __init__(self, electrode, in_channels, out_channels, Full_elec=False, Unsupervise=True):
        super(USRL, self).__init__()
        self.encoder = Encoder(electrode, in_channels, out_channels, Full_elec)
        self.softmax = torch.nn.Softmax(dim=1)
        self.activation = torch.nn.LeakyReLU()
        self.Full_elec = Full_elec
        self.Unsupervise = Unsupervise
        self.out_channels = out_channels
        self.electrode = electrode

    def forward(self, x):
        x = self.encoder(x)
        if not self.Unsupervise:
            if self.Full_elec:
                x = x.unsqueeze(dim=1)
            x = self.classification(x)
            x = self.softmax(x)
        return x

    def set_Unsupervised(self, device, Unsupervise):
        self.Unsupervise = Unsupervise
        print(self)
        if self.Full_elec:
            self.classification = nn.Sequential(
                torch.nn.utils.weight_norm(nn.Conv1d(1, 2, 3, 1)),
                nn.ReLU(),
                nn.MaxPool1d(2),
                torch.nn.utils.weight_norm(nn.Conv1d(2, 4, 3, 1)),
                nn.ReLU(),
                nn.MaxPool1d(2),
                torch.nn.utils.weight_norm(nn.Conv1d(4, 8, 3, 1)),
                nn.ReLU(),
                nn.MaxPool1d(2),
                torch.nn.Flatten(),
                nn.Dropout(0.5),
                torch.nn.utils.weight_norm(nn.Linear(656, 5)),
            ).to(device)
            # resnetEEGMV.Resnet50Encoder( 1, 5)

        else:
            self.classification = nn.Sequential(
                torch.nn.utils.weight_norm(nn.Conv1d(64, 16, 3)),
                nn.ReLU(),
                torch.nn.utils.weight_norm(nn.Conv1d(16, 4, 3)),
                nn.ReLU(),
                torch.nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(112, 5),
            ).to(device)

            # resnetEEG.Resnet50Encoder(64, 5)

