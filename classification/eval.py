import wandb
import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel

from utils import Observe


def validate(val_loader, model, prefix, epoch):
    """
    Run evaluation
    """
    # switch to evaluate mode
    model.eval()
    model_observe = Observe(prefix=prefix)
    criterion = torch.nn.CrossEntropyLoss()
    # print(val_loader.dataset[0])

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            img, trg = batch['img'].cuda(), batch['gt'].cuda()

            # compute output
            output = model(img)
            loss = criterion(output, trg)

            # measure accuracy and record loss
            model_observe.update(output, batch)

    wandb.log(model_observe.get_info_dict(), step=epoch)

    return model_observe
