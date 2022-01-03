import wandb
import torch.utils.data

from utils import Observe

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
        img += 0.1 * torch.randn(img.size()).cuda()

        # compute output
        output = model(img)
        loss = criterion(output, gt)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        model_observe.update(output, batch)

    wandb.log(model_observe.get_info_dict(), step=epoch)
    return model_observe
