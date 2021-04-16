# test file 

import torch
import torch.nn as nn
import time
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.Synapse import Synapse
from model.TransUNet import TransUNet
from utils.logging import AverageMeter, ProgressMeter
from tqdm import tqdm
from medpy import metric
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision
#import TransUnet_trained




torch.manual_seed(68)
np.random.seed(68)

dataset = Synapse(data_dir='dataset/Synapse/train_npz', mode='train')
train_loader = DataLoader(dataset=dataset, batch_size=3)

batch = next(iter(train_loader))
img_batch = batch['image']
label_batch = batch['label']

model = TransUNet()
model.load_state_dict(torch.load('TransUnet_trained'))

output = model(img_batch)
output = torch.argmax(torch.softmax(output, dim=1), dim=1)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
i = 0
ax1.imshow(img_batch[i][0], cmap='gray')
ax2.imshow(output[i], cmap='gnuplot')
ax3.imshow(label_batch[i][0], cmap='gnuplot')
plt.show()

