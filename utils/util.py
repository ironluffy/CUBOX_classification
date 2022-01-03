import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def get_optimizer(dataset, optim_name, model, lr):
    if dataset in ['cubox']:
        if optim_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-3)
        elif optim_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr)
        elif optim_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return optimizer


def get_lr_scheduler(dataset, sched, optimizer, epochs):
    if dataset == 'cubox':
        if sched == 'multi':
            lr_sched = lr_scheduler.MultiStepLR(optimizer, [int(0.6 * epochs), int(0.8 * epochs)], 0.1)
        elif sched == 'cos':
            lr_sched = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0001)
        elif sched == 'cos_warm':
            lr_sched = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0.0001)
        elif sched == 'none':
            lr_sched = None
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return lr_sched
