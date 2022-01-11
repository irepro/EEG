from UnsupervisedEncoder import *
from DataLoader import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from TripletSigmoidLoss import *

# batch size
batch_size = 16
learning_rate = 0.000001
epochs = 10

data_dir = "../USRLFOREEG/data"

trainEEG = Loader(data_dir + "/train/TIME_Sess01_sub01_train.npy", True)
trlblEEG = trainEEG.setLabel(data_dir + "/train/TIME_Sess01_sub01_trlbl.npy")

testEEG = Loader(data_dir + "/test/TIME_Sess01_sub01_test.npy", True)
tslblEEG = testEEG.setLabel(data_dir + "/test/TIME_Sess01_sub01_tslbl.npy")

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)
print("valLoader")
validLoader = DataLoader(testEEG, batch_size = batch_size, shuffle=True)

model = Encoder(1, 3)
criterion = TripletSigmoidLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

loss_tr = []
for epoch in range(epochs):
    loss_ep = 0
    for batch_idx, batch in enumerate(trainLoader):
        loss_batch = criterion.forward(batch, model, trainEEG.getallitem())
        optimizer.step()
        loss_ep += loss_batch
    
    scheduler.step()
    loss_tr.append(loss_ep)
    print("epoch : ",epoch, "   loss : ",loss_ep)

import os
from datetime import datetime

now = datetime.now()
date = now.strftime('%Y-%m-%d(%H:%M)')

savepath = "../USRLFOREEG/save_model/" + str(loss_tr[-1].item()) + ".pth"
torch.save(model.state_dict(), savepath)
