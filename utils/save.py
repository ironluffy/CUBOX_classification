import torch


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    if is_best:
        filename = filename.split(".")[0] + "_best." + filename.split(".")[1]
    torch.save(state, filename)
