from UnsupervisedEncoder import *
from DataLoader import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from TripletSigmoidLoss import *
from scipy.signal import butter, lfilter

data_dir = "../USRLFOREEG/data"
temp = np.load(data_dir + "/train/TIME_Sess01_sub01_train.npy")
temp = temp.transpose([2,0,1])
npy = torch.Tensor(temp[:,1,:])
npy_1 = torch.Tensor(BPFEEG(npy, 14, 30))
npy_2 = torch.Tensor(BPFEEG(npy[0,:], 14, 30))
print(npy_1[0,:])
print(npy_2)

def BPFEEG(eeg, low, high, fs = 512, order = 5):
    b, a = bandpass(low, high, fs, order)
    y = lfilter(b, a, eeg)
    return y

def bandpass(low, high, fs, order):
    nyq = 0.5*fs
    l = low/nyq
    h = high/nyq
    b,a = butter(order, [l,h], btype='band')
    return b,a
