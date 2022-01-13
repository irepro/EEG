import os
import numpy as np
from torch.utils.data import Dataset
import torch
from scipy.signal import butter, lfilter

class EEGLoader(Dataset):
    def __init__(self, data_dir, one_channel = True ,supervised = False, bpf = False):
        temp = np.load(data_dir)
        shape = temp.shape
        temp = temp[:, ::10, :]
        temp = temp.transpose([2,0,1])
        if one_channel:
            npy = torch.Tensor(temp[:,1,:])
            if bpf:
                npy = torch.Tensor(BPFEEG(npy, 14, 30))
                npy = npy.reshape([shape[2], 1, -1])
        else:
            if bpf:
                npy = torch.Tensor(BPFEEG(temp, 14, 30))
        self.x_data = npy
        self.supervised = supervised

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        if self.supervised:
            return self.x_data[index,:], self.y_data[index]
        else:
            return self.x_data[index,:]

    def getallitem(self):
        if self.supervised:
            return self.x_data, self.y_data
        else:
            return self.x_data

    def setLabel(self, data_dir):
        y_temp = np.load(data_dir)
        shape = y_temp.shape

        y_data = np.empty((shape[1], 1), dtype=np.float32)
        for i in range(shape[1]):
            y_data[i, :] = y_temp[1, i]
        self.y_data = y_data.squeeze()

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
