import torch


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    if is_best:
        filename = ".".join(filename.split(".")[:-1]) + "_best." + filename.split(".")[-1]
    torch.save(state, filename)
