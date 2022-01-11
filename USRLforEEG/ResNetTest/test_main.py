from net.UnsupervisedEncoder import *
from net.DataLoader import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from net.TripletSigmoidLoss import *
import keras_resnet.models

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

PATH = "../USRLFOREEG/save_model/134.9426727294922.pth"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint)

shape, classes = (32, 32, 3), 10

model = keras_resnet.models.ResNet50(x, classes=classes)

model.compile("adam", "categorical_crossentropy", ["accuracy"])

(training_x, training_y), (_, _) = keras.datasets.cifar10.load_data()

training_y = keras.utils.np_utils.to_categorical(training_y)

model.fit(training_x, training_y)