import os
from datetime import datetime
import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model import USRL
import utilLoader
import utils
from loss import TripletSigmoidLoss, TripletSigmoidLoss_MV
import torch

# batch size
batch_size = 3
learning_rate = 0.001
epochs = 5

os.environ["CUDA_VISIBLE_DEVICES"] = "3,7"
device = torch.device("cpu")
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#dataset 몇개를 사용할 것인지 결정 ex)1~4
idx = list(range(1,2))
tr, va, te = utils.load_dataset(idx).call(3)

# dataset loader
trainEEG = utilLoader.EEGLoader(tr, device, False)

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)

max_norm = 5
#if in_channels == 1: use one channel, in_channels == 62 : use 62 channel
in_channels = 1
#out_channels means the number of features of representation vector
out_channels = 32
electrode = 64
#Full_elec means whether you use all of electrodes or not, if true, then you will use all of electrodes
Full_elec = False

model = USRL.USRL(electrode, in_channels, out_channels, Full_elec).to(device)
#Custom Tripletloss

if Full_elec:
    criterion = TripletSigmoidLoss_MV.TripletSigmoidLoss(Kcount=5, scale_int=0.2)
else:
    criterion = TripletSigmoidLoss.TripletSigmoidLoss(Kcount=2, electrode = electrode, scale_int=1, sample_margin=100)
#use SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

# save epoch loss
loss_tr = []
loss_val=[]
for epoch in range(epochs):
    loss_ep = 0 # add batch loss in epoch
    for batch_idx, batch in enumerate(trainLoader):
        loss_batch = criterion.forward_SM(batch, model, trainEEG)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        loss_ep += loss_batch