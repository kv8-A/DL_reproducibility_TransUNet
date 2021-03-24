import torch

import time
import numpy as np
from collections import OrderedDict

import dataset
import model

from utils.logging import AverageMeter, ProgressMeter
from utils.config import parse_cli_overides

config = OrderedDict(
    # === DATASET ===
    train_size=1000,
    validation_size=500,
    test_size=500,

    # === OPTIMIZER ===
    batch_size=100,
    epochs=30,
    num_workers=8,
    optimizer="Adam",
    learning_rate=0.008,
)


def main():
    # Set the seed
    torch.manual_seed(68)
    np.random.seed(68)

    train_dataset = ... # TODO
    validation_dataset = ... # TODO

    # define the data loaders
    train_loader = ... # TODO

    val_loader = ... # TODO

    model = ... # TODO

    if torch.cuda.is_available():
        model = ... # TODO Push model to GPU
        print('Models pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    # Define optimizer
    optimizer = ... # TODO

    # Define loss
    criterion = ... # TODO

    # For storing the statistics
    metrics = {'train_loss': [],
               'train_acc': [],
               'val_loss': [],
               'val_acc': []}

    for epoch in range(config['epochs']):

        start = time.time()
        train_loss, train_acc = train_epoch(train_loader, model,
                                            criterion, optimizer, epoch)
        end = time.time()

        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        print('Epoch {} train loss: {:.4f}, acc: {:.4f}, time: {:.4f}'.format(epoch, train_loss, train_acc, end-start), flush=True)

        val_loss, val_acc = validate_epoch(val_loader, model,
                                        criterion, epoch)

        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        print('Epoch {} validation loss: {:.4f}, acc: {:.4f}'.format(epoch, val_loss, val_acc), flush=True)


def train_epoch(dataloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, loss_running, acc_running],
        prefix="Train, epoch: [{}]".format(epoch))

    start = time.time()

    model.train()

    with torch.set_grad_enabled(True):
        # Iterate over data.
        for epoch_step, (input, label) in enumerate(dataloader):

            # TODO Training procedure


            # Loss
            loss = ...  # TODO
            loss_running.update(...)

            # Top-1 accuracy
            accuracy = ... # TODO
            acc_running.update(...)

            progress.display(epoch_step)

            # Measure time
            batch_time.update(time.time() - start)
            start = time.time()

    return loss_running.avg, acc_running.avg


def validate_epoch(dataloader, model, criterion, epoch):
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
    config = parse_cli_overides(config)
    print("########## Printing Full Config ##############", flush=True)
    print(config, flush=True)

    main()
