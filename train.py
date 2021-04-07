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

"""
From [paper 4.1]:

Metrics:
    - dice score DSC for each organ [%]
    - average dice score DSC [%]
    - average hausdorff distance [mm]

8 organs (=> 9 classes)

Training: 18 cases - 2211 total slices NOTE: sais 2212, but we recieved 2211

Testing : 12 cases

"""

config = OrderedDict(
    # === DATASET ===
    n_classes=9,

    # === OPTIMIZER ===
    batch_size=24, # [paper 4.2]
    max_iterations = 14000, # [paper 4.2]
    # epochs=30,
    epochs=5, # for debugging
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
    # for epoch in tqdm(range(config['epochs'])):
    for epoch in tqdm(range(config['epochs']), desc='Epochs'):

        # Train on data
        train_loss = train(epoch, train_loader, model, optimizer, criterion)

        # Test on data
        test_loss = test(epoch, train_loader, model)

        # Metrics
        # end = time.time()
        writer.add_scalars("Loss", {"Train": train_loss}, epoch)

        # val_loss, val_acc = validate_epoch(val_loader, model,
        #                                 criterion, epoch)


def train(epoch, train_loader, model, optimizer, criterion):
    """ 
    Trains network for one epoch in batches
    """

    # Metrics
    # start = time.time()
    # batch_time = AverageMeter('Time', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, loss_running],
    #     prefix="Train, epoch: [{}]".format(epoch)
    # )

    # Switch to training mode
    model.train()
    torch.set_grad_enabled(True)

    # Batches loop
    for i, batch in enumerate(train_loader):

        # Get batch data
        input_batch = batch['image']
        label_batch = batch['label']
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Foreward pass
        output = model(input_batch)

        # Loss
        # output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        label_batch = label_batch.squeeze(1)
        loss = criterion(output, label_batch.long())

        # Backwards pass
        loss.backward()
        optimizer.step()

        # TODO: do some lr scheduling?

        # Metrics
        loss_running.update(loss.item())
        # progress.display(i)
        # batch_time.update(time.time() - start)
        # start = time.time()

    return loss_running.avg

def test(epoch, test_loader, model):

    # Metrics
    loss_running = AverageMeter('Loss', ':.4e')

    # Switch to testing mode
    model.eval()
    torch.set_grad_enabled(False)

    # Batches loop
    for i, batch in enumerate(test_loader):

        # Get batch data
        input_batch = batch['image']
        label_batch = batch['label']

        print(input_batch.shape)

        # Foreward pass
        output = model(input_batch)

        # Loss
        label_batch = label_batch.squeeze(1)
        # loss = criterion(output, label_batch.long())

    return 0


if __name__ == "__main__":
    main()
