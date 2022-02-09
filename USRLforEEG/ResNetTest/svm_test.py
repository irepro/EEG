import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
from net import EEGLoader
from model import UnsupervisedEncoder, resnetEEG
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn import svm

def accuracy_check(label, pred):
    prediction = np.around(pred)

    compare = np.equal(label, prediction)
    accuracy = np.sum(compare.tolist()) / len(compare.tolist())
    return accuracy

data_dir = "../USRLFOREEG/data"

trainEEG = EEGLoader.EEGLoader(data_dir + "/train/TIME_Sess01_sub01_train.npy", num_channel = 62, supervised = True, bpf = True)
trlblEEG = trainEEG.setLabel(data_dir + "/train/TIME_Sess01_sub01_trlbl.npy")

testEEG = EEGLoader.EEGLoader(data_dir + "/test/TIME_Sess01_sub01_test.npy", num_channel = 62, supervised = True, bpf = True)
tslblEEG = testEEG.setLabel(data_dir + "/test/TIME_Sess01_sub01_tslbl.npy")

'''in_channels = 1
#out_channels means the number of features of representation vector 
out_channels = 32
electrode = 62

represent_encoder = UnsupervisedEncoder.Encoder(electrode, in_channels, out_channels)'''

PATH = "../USRLFOREEG/save_model/02051302ch72loss2.pth"
represent_encoder = torch.load(PATH, map_location=torch.device('cpu'))


x_tr, y_tr = trainEEG.getallitem()
x_te, y_te = testEEG.getallitem()
rptr_batch = represent_encoder.forward(x_tr)
rptr_batch = rptr_batch.reshape([100, -1])


clf = svm.SVC(max_iter = 10000)              # 학습 반복횟수 10000
clf.fit(rptr_batch.detach().numpy(), y_tr)

rpte_batch = represent_encoder.forward(x_te)
rpte_batch = rpte_batch.reshape([100, -1])
y_pred = clf.predict(rpte_batch.detach().numpy())

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

epoch_acc = accuracy_check(y_te, y_pred)

print("val acc : ", epoch_acc)

label=[0, 1] # 라벨 설정
plot = confusion_matrix(y_te, y_pred)
print(plot)