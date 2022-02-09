import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from net import EEGLoader
from model import MSNN
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import tensorflow as tf 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
from datetime import datetime

def gradient(model, inputs, labels):
    with tf.GradientTape() as tape:
        y_hat, _ = model(inputs)
        loss = tf.keras.losses.binary_crossentropy(labels, y_hat)
        
    grad = tape.gradient(loss, model.trainable_variables)
    return loss, grad

def accuracy_check(label, pred):
    compare = np.equal(label, pred)
    accuracy = np.sum(compare.tolist())
    return accuracy



data_dir = "../USRLFOREEG/data"

trainEEG = EEGLoader.EEGLoader(data_dir + "/train/TIME_Sess01_sub01_train.npy", downsampling = 10, num_channel = 62, supervised = True, bpf = True)
trlblEEG = trainEEG.setLabel(data_dir + "/train/TIME_Sess01_sub01_trlbl.npy", False)

testEEG = EEGLoader.EEGLoader(data_dir + "/test/TIME_Sess01_sub01_test.npy", downsampling = 10, num_channel = 62, supervised = True, bpf = True)
tslblEEG = testEEG.setLabel(data_dir + "/test/TIME_Sess01_sub01_tslbl.npy")

batch_size = 8
learning_rate = 0.01
epochs = 30

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)
print("valLoader")
validLoader = DataLoader(testEEG, batch_size = batch_size, shuffle=True)

cla_model = MSNN.MSNN(62)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

total_acc = []
total_loss = []
for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0
    for batch, (inputs, labels) in enumerate(trainLoader):
        inputs = torch.unsqueeze(inputs, 3)
        loss, grads = gradient(cla_model, inputs.numpy(),  labels)
        optimizer.apply_gradients(zip(grads, cla_model.trainable_variables))

        epoch_loss += np.mean(loss)  

    total_acc.append(epoch_acc/batch_size)
    total_loss.append(epoch_loss)

    print("epoch", epoch + 1, "train loss : ", epoch_loss)


inputs, labels = testEEG.getallitem()
inputs = torch.unsqueeze(inputs, 3)
task_pred, _ = cla_model(inputs.numpy())

pred = np.argmax(task_pred, axis=1)
epoch_acc = accuracy_check(labels, pred)

print("val acc : ", epoch_acc/len(labels))

label=[0, 1] # 라벨 설정
plot = confusion_matrix(labels, pred)
print(plot)
score = str(f1_score(labels, pred))
print("f1 acc : ", score)
  


'''
batch_size = 8
learning_rate = 0.001
epochs = 10

print("tainLoader")
trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)
print("valLoader")
validLoader = DataLoader(testEEG, batch_size = batch_size, shuffle=True)

#PATH = "../USRLFOREEG/save_model/01221701ch32loss1.pth"
#represent_encoder = torch.load(PATH, map_location=torch.device('cpu'))

cla_model = MSNN.MSNN_rpts(62)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

total_loss = []
for epoch in range(epochs):
    epoch_loss = 0
    for batch, (inputs, labels) in enumerate(trainLoader):
        
        #rpt_batch = represent_encoder.forward(inputs)
        #rpt_batch = torch.unsqueeze(rpt_batch, 3)
        
        #loss, grads = gradient(cla_model, rpt_batch.detach().numpy(), labels)
        loss, grads = gradient(cla_model, inputs, labels)
        optimizer.apply_gradients(zip(grads, cla_model.trainable_variables))

        epoch_loss += np.mean(loss)  

    total_loss.append(epoch_loss)

    print("epoch", epoch + 1, "train loss : ", epoch_loss)


inputs, labels = testEEG.getallitem()
#rpt_batch = represent_encoder.forward(inputs)
#rpt_batch = torch.unsqueeze(rpt_batch, 3)

task_pred, _ = cla_model(inputs)

pred = np.argmax(task_pred, axis=1)
epoch_acc = accuracy_check(labels, pred)

print("val acc : ", epoch_acc)
'''
    

