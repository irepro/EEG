import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
import utils
import utilLoader
from model import USRL
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def accuracy_check(label, pred):
    prediction = np.argmax(pred, axis=1)

    compare = np.equal(label, prediction)
    accuracy = np.sum(compare.tolist()) / len(compare.tolist())

    print(prediction)
    print(labels)

    return accuracy, prediction

batch_size = 4
learning_rate = 0.0001
epochs = 5

idx = list(range(1,2))
tr, va, te = utils.load_dataset(idx).call(5)

# dataset loader
trainEEG = utilLoader.EEGLoader(tr, True)

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)

#represent_encoder = UnsupervisedEncoder.Encoder(electrode, in_channels, out_channels)

name = "172302c32l23elecF"
PATH = "../USRL/save_model/"+name+".pth"
model = torch.load(PATH, map_location=torch.device('cpu'))
model.set_Unsupervised(False)

#max_norm = 5
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
        loss = CrossEL(outputs, labels)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        epoch_loss += loss.clone().item()

        loss = 0
        torch.cuda.empty_cache()
    
    total_loss.append(epoch_loss)
    scheduler.step()

    inputs, labels = va

    outputs = model.forward(torch.Tensor(inputs))
    loss_va = CrossEL(outputs, torch.Tensor(labels))

    print("epoch", epoch + 1, "train loss : ", epoch_loss, "val loss:", loss_va.item())

inputs, labels = te
labels = np.argmax(labels, axis=1)

outputs = model.forward(torch.Tensor(inputs))
epoch_acc, pred = accuracy_check(labels, outputs.detach().numpy())

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
