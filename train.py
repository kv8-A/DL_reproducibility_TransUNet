import torch
import torch.nn as nn
import time
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader

from dataset import Synapse
from model import TransUNet
from utils.logging import AverageMeter, ProgressMeter

config = OrderedDict(
    # === DATASET ===
    # train_size=1000,
    # validation_size=500,
    # test_size=500,
    n_classes=9,

    # === OPTIMIZER ===
    batch_size=24, # [paper 4.2]
    max_iterations = 14000, # [paper 4.2]
    # epochs=30,
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
    dataset = Synapse(data_dir='dataset/Synapse/train_npz', mode='train')
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'])

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
    criterion = nn.CrossEntropyLoss() # NOTE: CE assumed, loss function not mentioned in paper

    # For storing the statistics
    metrics = {'train_loss': [],
               'train_acc': [],
               'val_loss': [],
               'val_acc': []}

    # Epoch loop
    for epoch in range(config['epochs']):
        # Logging
        start = time.time()
        batch_time = AverageMeter('Time', ':6.3f')
        loss_running = AverageMeter('Loss', ':.4e')
        acc_running = AverageMeter('Accuracy', ':.3f')
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, loss_running, acc_running],
            prefix="Train, epoch: [{}]".format(epoch))
            
        # Switch to train mode
        model.train()

        # Batches loop
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Get batch data
            input_batch = batch['image']
            label_batch = batch['label']

            # Foreward pass
            output = model(input_batch)

            # Loss
            loss = criterion(output, label_batch)
            loss_running.update(loss.item())

            # Backwards pass
            loss.backward()
            optimizer.step()

            # Accuracy
            accuracy = ... # TODO
            acc_running.update(...)

            # Logging
            progress.display(i)
            batch_time.update(time.time() - start)
            start = time.time()

        # Logging
        end = time.time()
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        print('Epoch {} train loss: {:.4f}, acc: {:.4f}, time: {:.4f}'.format(epoch, train_loss, train_acc, end-start), flush=True)

        val_loss, val_acc = validate_epoch(val_loader, model,
                                        criterion, epoch)

        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        print('Epoch {} validation loss: {:.4f}, acc: {:.4f}'.format(epoch, val_loss, val_acc), flush=True)


def train_epoch(dataloader, model, loss, optimizer, epoch):
    # # Logging
    # batch_time = AverageMeter('Time', ':6.3f')
    # loss_running = AverageMeter('Loss', ':.4e')
    # acc_running = AverageMeter('Accuracy', ':.3f')
    # progress = ProgressMeter(
    #     len(dataloader),
    #     [batch_time, loss_running, acc_running],
    #     prefix="Train, epoch: [{}]".format(epoch))
    # start = time.time()

    # # Train
    # model.train()

    # with torch.set_grad_enabled(True):
        # Iterate over data.
        # for epoch_step, (input, label) in enumerate(dataloader):

            # # Loss
            # loss = ...  # TODO
            # loss_running.update(...)

            # Top-1 accuracy
            # accuracy = ... # TODO
            # acc_running.update(...)

            # progress.display(epoch_step)

            # Measure time
            # batch_time.update(time.time() - start)
            # start = time.time()

    return loss_running.avg, acc_running.avg


def validate_epoch(dataloader, model, loss, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, loss_running, acc_running],
        prefix="Validation, epoch: [{}]".format(epoch))

    start = time.time()

    model.eval()

    # TODO Implement validation procedure similarly to the training procedure

    return loss_running.avg, acc_running.avg


if __name__ == "__main__":
    main()
