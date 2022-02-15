import os
import numpy as np
from torch.utils.data import Dataset
import torch
from scipy.signal import butter, lfilter

class EEGLoader(Dataset):
    def __init__(self, data_dir, downsampling = 10, num_channel = 62 ,supervised = False, bpf = False):
        temp = np.load(data_dir)
        shape = temp.shape
        temp = temp[:, ::downsampling, :]
        temp = temp.transpose([2,0,1])
        if num_channel == 1:
            npy = torch.Tensor(temp[:,0,:])
            self.electrode_index = [0]
            if bpf:
                npy = torch.Tensor(BPFEEG(npy, 14, 30))
                npy = npy.reshape([shape[2], 1, -1])
        elif num_channel == 62:
            npy = torch.Tensor(temp[:,:,:])
            self.electrode_index = ['all']
            if bpf:
                npy = torch.Tensor(BPFEEG(npy, 14, 30))
        else:
            index = np.random.randint(
                    shape[0], size=num_channel
                )
            self.electrode_index = index
            npy = temp[:,index,:]
            if bpf:
                npy = torch.Tensor(BPFEEG(temp, 14, 30))
        self.x_data = npy
        self.supervised = supervised

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        if self.supervised:
            if self.one_dim:
                return self.x_data[index,:], self.y_data[index]
            else:
                return self.x_data[index,:], self.y_data[index,:]
        else:
            return self.x_data[index,:]

    def getallitem(self):
        if self.supervised:
            return self.x_data, self.y_data
        else:
            return self.x_data

    def setLabel(self, data_dir, one_dim = True):
        self.one_dim = one_dim
        y_temp = np.load(data_dir).astype(np.float32)
        shape = y_temp.shape
        if one_dim:
            y_data = np.empty((shape[1], 1), dtype=np.float32)
            for i in range(shape[1]):
                y_data[i, :] = y_temp[1, i]
            self.y_data = y_data.squeeze()
        else:
            self.y_data = y_temp.T

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
