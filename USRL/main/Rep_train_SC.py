
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
batch_size = 4
learning_rate = 0.00000000001
epochs = 15

idx = list(range(1,4))
tr, va, te = utils.load_dataset(idx).call(3)

# dataset loader
trainEEG = utilLoader.EEGLoader(tr, False)

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)

#if in_channels == 1: use one channel, in_channels == 62 : use 62 channel
in_channels = 1
#out_channels means the number of features of representation vector 
out_channels = 256
electrode = 64
Full_elec = True

model = USRL.USRL(electrode, in_channels, out_channels, Full_elec)
#Custom Tripletloss

if Full_elec:
    criterion = TripletSigmoidLoss_MV.TripletSigmoidLoss(Kcount=5, scale_int=0.2)     
else:
    criterion = TripletSigmoidLoss.TripletSigmoidLoss(Kcount=3, electrode = electrode, scale_int=1)
#use SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = StepLR(optimizer, step_size=6, gamma=0.5)

# save epoch loss
loss_tr = []
loss_val=[] 
for epoch in range(epochs):
    loss_ep = 0 # add batch loss in epoch
    for batch_idx, batch in enumerate(trainLoader):
        loss_batch = criterion.forward(batch, model, trainEEG.getallitem())
        optimizer.step()
        loss_ep += loss_batch
    
    loss_v = criterion.get_valloss(model, torch.Tensor(va[0]))
    loss_tr.append(loss_ep.item()/1000)
    loss_val.append(loss_v.item())
    #scheduler.step()
    print("epoch : ",epoch, "   train loss : ",loss_ep.item(),"    val loss : ", loss_v.item())

loss_te = criterion.get_valloss(model, torch.Tensor(te[0]))
print("test loss : ", loss_te.item())

now = datetime.now()
date = now.strftime('%d%H%m')
if Full_elec:
    fe = "T"
else:
    fe = "F"
savepath = "../USRL/save_model/"+date+ "c" + str(out_channels) + "l" +str(int(loss_val[-1])) +"elec"+ fe + ".pth"
torch.save(model, savepath)

plt.plot(range(epochs), loss_tr, label='Loss', color='red')
plt.plot(range(epochs), loss_val, label='Loss', color='blue')

plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

