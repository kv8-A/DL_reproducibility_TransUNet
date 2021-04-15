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
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision

"""
From [paper 4.1]:

8 organs (=> 9 classes)

Training: 18 cases - 2211 total slices NOTE: sais 2212, but we recieved 2211

Testing : 12 cases

"""

config = OrderedDict(
    # === DATASET ===
    n_classes=9,

    # === OPTIMIZER ===
    batch_size=10, # [paper 4.2]
    max_iterations = 14000, # [paper 4.2]
    # epochs=30, # default assumption
    epochs=30, # for debugging
    # num_workers=8,
    learning_rate=0.01, # [paper 4.2]
    momentum=0.01, # [paper 4.2]
    weight_decay=0.0001, # [paper 4.2]
)

def main():
    # Set the seed
    torch.manual_seed(68)
    np.random.seed(68)

    # Define the data
    train_dataset = Synapse(data_dir='dataset/Synapse/train_npz', mode='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'])
    test_dataset = Synapse(data_dir='dataset/Synapse/test_vol_h5', mode='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1) # batch size 1 because the volumes are batches themselves

    # Define network
    model = TransUNet()

    # Push model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print('Models pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    
    # Define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    # Define loss
    criterion = nn.CrossEntropyLoss() # NOTE: CE assumed, but loss function not mentioned in paper

    # Create Tensorboard writer
    writer = SummaryWriter()

    # Epoch loop
    for epoch in tqdm(range(config['epochs']), desc='Epochs'):

        # Train on data
        train_loss = train(epoch, train_loader, model, optimizer, criterion)

        # Metrics
        writer.add_scalars("Loss", {"Train": train_loss}, epoch)

    torch.save(model.state_dict(), 'TransUnet_trained')

def train(epoch, train_loader, model, optimizer, criterion):
    """ 
    Trains network for one epoch in batches
    """

    # Metrics
    loss_running = AverageMeter('Loss', ':.4e')

    # Switch to training mode
    model.train()
    torch.set_grad_enabled(True)

    # Batches loop
    for i, batch in enumerate(train_loader):

        # Get batch data
        img_batch = batch['image'].cuda()
        label_batch = batch['label'].cuda()
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(img_batch)

        # Loss
        # output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        label_batch = label_batch.squeeze(1) # remove the 1 channel dim, as required by the criterion
        loss = criterion(output, label_batch.long())

        # Backwards pass
        loss.backward()
        optimizer.step()

        # Metrics
        loss_running.update(loss.item())

    return loss_running.avg

if __name__ == "__main__":
    main()
    