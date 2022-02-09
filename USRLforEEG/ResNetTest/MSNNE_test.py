import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from net import EEGLoader
from model import rptsMSNN, resnetEEGnoConnect, simpleClass
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import tensorflow as tf

def accuracy_check(label, pred):
    prediction = np.around(pred)

    compare = np.equal(label, prediction)
    accuracy = np.sum(compare.tolist()) / len(compare.tolist())
    return accuracy

batch_size = 8
learning_rate = 0.0001
epochs = 20

data_dir = "../USRLFOREEG/data"

trainEEG = EEGLoader.EEGLoader(data_dir + "/train/TIME_Sess01_sub01_train.npy", num_channel = 62, supervised = True, bpf = True)
trlblEEG = trainEEG.setLabel(data_dir + "/train/TIME_Sess01_sub01_trlbl.npy", False)

testEEG = EEGLoader.EEGLoader(data_dir + "/test/TIME_Sess01_sub01_test.npy", num_channel = 62, supervised = True, bpf = True)
tslblEEG = testEEG.setLabel(data_dir + "/test/TIME_Sess01_sub01_tslbl.npy", False)

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)
print("valLoader")
validLoader = DataLoader(testEEG, batch_size = batch_size, shuffle=True)

in_channels = 1
#out_channels means the number of features of representation vector 
out_channels = 96
electrode = 62

#represent_encoder = UnsupervisedEncoder.Encoder(electrode, in_channels, out_channels)


PATH = "../USRLFOREEG/save_model/01241601ch32loss2.pth"
represent_encoder = torch.load(PATH, map_location=torch.device('cpu'))
#represent_encoder.load_state_dict(checkpoint)

resnetClasification = rptsMSNN.Resnet50Encoder(62,2)
#resnetClasification = simpleClass.Net()

max_norm = 5
#BCEloss = torch.nn.BCELoss()
CrossEL=torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(resnetClasification.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(resnetClasification.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

total_loss = []
for epoch in range(epochs):

    epoch_loss = 0
    for batch, (inputs, labels) in enumerate(trainLoader):
        #mini_size = inputs.size(0)
        rpt_batch = represent_encoder.forward(inputs)
        #rpt_batch = []
        #for i in range(mini_size):
        #    input = inputs[i,:,:]
        #    rpt_batch.append(represent_encoder.forward(input)
        #rpt_batch = torch.stack(rpt_batch)
        optimizer.zero_grad()

        outputs = resnetClasification.forward(rpt_batch)
        outputs = outputs.squeeze()
        loss = CrossEL(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(resnetClasification.parameters(), max_norm)
        optimizer.step()

        epoch_loss += loss.clone().item()

        loss = 0
        torch.cuda.empty_cache()
    
    total_loss.append(epoch_loss)
    #scheduler.step()

    print("epoch", epoch + 1, "train loss : ", epoch_loss)

inputs, labels = testEEG.getallitem()
labels = np.argmax(labels, axis=1)

rpt_batch = represent_encoder.forward(inputs)
#test_size = inputs.size(0)
#rpt_batch = []
#for i in range(test_size):
#    input = inputs[i,:,:]
# #   rpt_batch.append(represent_encoder.forward(input))
#rpt_batch = torch.stack(rpt_batch)

task_pred = resnetClasification.forward(rpt_batch)

pred = np.argmax(task_pred.detach().numpy(), axis=1)
epoch_acc = accuracy_check(labels, pred)

print("val acc : ", epoch_acc)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

label=[0, 1] # 라벨 설정
plot = confusion_matrix(labels, pred)
print(plot)
print("f1 acc : ", f1_score(labels, pred))