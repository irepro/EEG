import os
import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model import MSNNEncoder
from EEGLoader import *
from loss import TripletSigmoidLoss_MSNNE

# batch size
batch_size = 8
learning_rate = 0.00000001
epochs = 20

# dataset path
data_dir = "../USRLFOREEG/data"

# dataset loader
trainEEG = EEGLoader(data_dir + "/train/TIME_Sess01_sub01_train.npy", num_channel = 62, supervised = False, bpf = True)
trlblEEG = trainEEG.setLabel(data_dir + "/train/TIME_Sess01_sub01_trlbl.npy")

testEEG = EEGLoader(data_dir + "/test/TIME_Sess01_sub01_test.npy", num_channel = 62, supervised = False, bpf = True)
tslblEEG = testEEG.setLabel(data_dir + "/test/TIME_Sess01_sub01_tslbl.npy")

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)
print("valLoader")
validLoader = DataLoader(testEEG, batch_size = batch_size, shuffle=True)

#if in_channels == 1: use one channel, in_channels == 62 : use 62 channel
in_channels = 1
#out_channels means the number of features of representation vector 
out_channels = 32
electrode = 62

#model = UnsupervisedEncoder_batch.Encoder(electrode, in_channels, out_channels)
model = MSNNEncoder.Encoder(electrode, in_channels, out_channels)
'''PATH = "../USRLFOREEG/save_model/01191501ch128loss3.pth"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint)'''
#Custom Tripletloss

criterion = TripletSigmoidLoss_MSNNE.TripletSigmoidLoss(Kcount=5, electrode = 62, scale_int=0.2) 

#use SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# save epoch loss
loss_tr = []
loss_val=[] 
for epoch in range(epochs):
    loss_ep = 0 # add batch loss in epoch
    for batch_idx, batch in enumerate(trainLoader):
        loss_batch = criterion.forward(batch, model, trainEEG.getallitem())
        optimizer.step()
        loss_ep += loss_batch
    
    loss_v = criterion.get_valloss(model, testEEG.getallitem())
    loss_tr.append(loss_ep.item()/1000)
    loss_val.append(loss_v.item())
    scheduler.step()
    print("epoch : ",epoch, "   train loss : ",loss_ep.item(),"    val loss : ", loss_v.item())

import os
from datetime import datetime

now = datetime.now()
date = now.strftime('%m%d%H%m')

savepath = "../USRLFOREEG/save_model/"+date+ "ch" + str(out_channels) + "loss"+str(int(loss_val[-1])) + ".pth"
torch.save(model, savepath)

import matplotlib.pyplot as plt

plt.plot(range(epochs), loss_tr, label='Loss', color='red')
plt.plot(range(epochs), loss_val, label='Loss', color='blue')

plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

