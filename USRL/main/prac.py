import os
import sys
import utils 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import utilLoader

idx = list(range(1,2))
tr, va, te = utils.load_dataset(idx).call(5)

batch_size = 4

print("tainLoader")

trainEEG = utilLoader.EEGLoader(tr, False)

trainLoader = DataLoader(trainEEG, batch_size = batch_size, shuffle=True)
for batch, inputs in enumerate(trainLoader):
    print(va[0])

    print(inputs.size)

