import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from main import EEGLoader
from model import USRL
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def accuracy_check(label, pred):
    prediction = np.around(pred)

    compare = np.equal(label, prediction)
    accuracy = np.sum(compare.tolist()) / len(compare.tolist())
    return accuracy

batch_size = 4
learning_rate = 0.0001
epochs = 40

data_dir = "../USRLFOREEG/data"

trainEEG = EEGLoader.EEGLoader(data_dir + "/train/TIME_Sess01_sub01_train.npy", num_channel = 62, supervised = True, bpf = True)
trlblEEG = trainEEG.setLabel(data_dir + "/train/TIME_Sess01_sub01_trlbl.npy", False)

testEEG = EEGLoader.EEGLoader(data_dir + "/test/TIME_Sess01_sub01_test.npy", num_channel = 62, supervised = True, bpf = True)
tslblEEG = testEEG.setLabel(data_dir + "/test/TIME_Sess01_sub01_tslbl.npy", False)

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)
print("valLoader")
validLoader = DataLoader(testEEG, batch_size = batch_size, shuffle=True)

#represent_encoder = UnsupervisedEncoder.Encoder(electrode, in_channels, out_channels)

name = "151902c512l12elecT"
PATH = "../USRL/save_model/"+name+".pth"
model = torch.load(PATH, map_location=torch.device('cpu'))
model.set_Unsupervised(False)

max_norm = 5
#BCEloss = torch.nn.BCELoss()
CrossEL=torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

total_loss = []
for epoch in range(epochs):

    epoch_loss = 0
    for batch, (inputs, labels) in enumerate(trainLoader):
        optimizer.zero_grad()

        outputs = model.forward(inputs)
        outputs = outputs.squeeze()
        loss = CrossEL(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        epoch_loss += loss.clone().item()

        loss = 0
        torch.cuda.empty_cache()
    
    total_loss.append(epoch_loss)
    scheduler.step()

    print("epoch", epoch + 1, "train loss : ", epoch_loss)

inputs, labels = testEEG.getallitem()
labels = np.argmax(labels, axis=1)

outputs = model.forward(inputs)
pred = np.argmax(outputs.detach().numpy(), axis=1)
epoch_acc = accuracy_check(labels, pred)
print(pred)
print(labels)

print("val acc : ", epoch_acc)

label=[0, 1] # 라벨 설정
plot = confusion_matrix(labels, pred)
print(plot)
score = str(f1_score(labels, pred))
print("f1 acc : ", score)

plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

savepath = "../USRLFOREEG/save_res/"+ "ch" + name[2:12] + "f1"+score[2:4] + "acc" + str(epoch_acc[2:]) + ".pth"
torch.save(model, savepath)
