import wandb
import torch.utils.data

from utils import Observe
from dataset.transforms import get_transform

method_hyperparams = dict()


def train(train_loader, model, optimizer, epoch, args, epoch_storage):
    """
        Run one train epoch
    """

    # switch to train mode
    model.train()
    model_observe = Observe(prefix='Train')
    criterion = torch.nn.CrossEntropyLoss()

    for i, batch in enumerate(train_loader):
        # measure data loading time
        optimizer.zero_grad()
        img, gt = batch['img'].cuda(), batch['gt'].cuda()

        # compute output
        output = model(img)
        loss = criterion(output, gt)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        model_observe.update(output, batch)

    wandb.log(model_observe.get_info_dict(), step=epoch)
    return model_observe


def train_mixed_label(train_loader, model, optimizer, epoch, args, epoch_storage):
    """
        Run one train epoch
    """

    # switch to train mode
    model.train()
    model_observe = Observe(prefix='Train')
    criterion = torch.nn.CrossEntropyLoss()

    # mixing augmenatation
    mixer, _ = get_transform(args.mixed_transform)

    for i, batch in enumerate(train_loader):
        # measure data loading time
        optimizer.zero_grad()
        img, gt, gt_mixed, mixed_prop = mixer(batch)

        # compute output
        output = model(img)
        loss = criterion(output, gt) * mixed_prop + criterion(output, gt_mixed) * (1. - mixed_prop)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        model_observe.update(output, batch)

    wandb.log(model_observe.get_info_dict(), step=epoch)
    return model_observe
