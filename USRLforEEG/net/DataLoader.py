import os
import numpy as np
from torch.utils.data import Dataset
import torch
from scipy.signal import butter, lfilter

class Loader(Dataset):
    def __init__(self, data_dir, bpf = False):
        temp = np.load(data_dir)
        temp = temp[:, ::10, :]
        temp = temp.transpose([2,0,1])
        npy = torch.Tensor(temp[:,1,:])
        if bpf:
            npy = torch.Tensor(BPFEEG(npy, 14, 30))
        self.x_data = npy

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index,:]

    def getallitem(self):
        return self.x_data

    def setLabel(self, data_dir):
        y_data = np.load(data_dir)
        self.y_data = y_data

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
