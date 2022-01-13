from UnsupervisedEncoder import *
from EEGLoader import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from TripletSigmoidLoss import *

# batch size
batch_size = 16
learning_rate = 0.000001
epochs = 30

data_dir = "../USRLFOREEG/data"

trainEEG = EEGLoader(data_dir + "/train/TIME_Sess01_sub01_train.npy", one_channel = False, supervised = False, bpf = True)
trlblEEG = trainEEG.setLabel(data_dir + "/train/TIME_Sess01_sub01_trlbl.npy")

testEEG = EEGLoader(data_dir + "/test/TIME_Sess01_sub01_test.npy", one_channel = False, supervised = False, bpf = True)
tslblEEG = testEEG.setLabel(data_dir + "/test/TIME_Sess01_sub01_tslbl.npy")

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)
print("valLoader")
validLoader = DataLoader(testEEG, batch_size = batch_size, shuffle=True)

in_channels = 62
out_channels = 64

model = Encoder(in_channels, out_channels)
criterion = TripletSigmoidLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

loss_tr = []
loss_val=[]
for epoch in range(epochs):
    loss_ep = 0
    for batch_idx, batch in enumerate(trainLoader):
        loss_batch = criterion.forward(batch, model, trainEEG.getallitem())
        optimizer.step()
        loss_ep += loss_batch
    
    loss_v = criterion.get_valloss(model, testEEG.getallitem())
    scheduler.step()
    loss_tr.append(loss_ep.item())
    loss_val.append(loss_v.item())
    print("epoch : ",epoch, "   train loss : ",loss_ep.item(),"    val loss : ", loss_v.item())

import os
from datetime import datetime

now = datetime.now()
date = now.strftime('%Y-%m-%d(%H:%M)')

savepath = "../USRLFOREEG/save_model/" + str(in_channels) + "to" + str(out_channels) + "loss"+str(int(loss_tr[-1])) + ".pth"
torch.save(model.state_dict(), savepath)

import matplotlib.pyplot as plt

plt.plot(range(epochs), loss_tr, label='Loss', color='red')
plt.plot(range(epochs), loss_val, label='Loss', color='blue')

plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

