import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from net import EEGLoader, UnsupervisedEncoder, resnetEEG
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

def accuracy_check(label, pred):
    prediction = np.around(pred)

    compare = np.equal(label, prediction)
    accuracy = np.sum(compare.tolist()) / len(compare.tolist())
    return accuracy

batch_size = 16
learning_rate = 0.000001
epochs = 10

data_dir = "../USRLFOREEG/data"

trainEEG = EEGLoader.EEGLoader(data_dir + "/train/TIME_Sess01_sub01_train.npy", one_channel = False, supervised = True, bpf = True)
trlblEEG = trainEEG.setLabel(data_dir + "/train/TIME_Sess01_sub01_trlbl.npy")

testEEG = EEGLoader.EEGLoader(data_dir + "/test/TIME_Sess01_sub01_test.npy", one_channel = False, supervised = True, bpf = True)
tslblEEG = testEEG.setLabel(data_dir + "/test/TIME_Sess01_sub01_tslbl.npy")

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)
print("valLoader")
validLoader = DataLoader(testEEG, batch_size = batch_size, shuffle=True)

represent_encoder = UnsupervisedEncoder.Encoder(62, 64)

PATH = "../USRLFOREEG/save_model/62to64loss550.pth"
checkpoint = torch.load(PATH)
represent_encoder.load_state_dict(checkpoint)

resnetClasification = resnetEEG.Resnet50Encoder(1,2)
BCEloss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(resnetClasification.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

total_acc = []
total_loss = []
for epoch in range(epochs):
    resnetClasification.eval()

    epoch_loss = 0
    epoch_acc = 0
    for batch, (inputs, labels) in enumerate(trainLoader):
        ##########################################
        ############# fill in here -> train
        ####### Hint :
        rpt_batch = represent_encoder.forward(inputs)
        optimizer.zero_grad()

        outputs = resnetClasification.forward(rpt_batch)
        outputs = outputs.squeeze()
        loss = BCEloss(outputs, labels)

        epoch_loss += loss.clone().item()
        epoch_acc += accuracy_check(labels, outputs.clone().detach())

        loss.backward()
        optimizer.step()
    
    total_acc.append(epoch_acc/batch_size)
    total_loss.append(epoch_loss)
    scheduler.step()

    print("epoch", epoch + 1, "train loss : ", epoch_loss, "train acc : ", epoch_acc/batch_size)

