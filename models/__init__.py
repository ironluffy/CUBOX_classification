import torch.nn as nn
from torchvision.models.resnet import resnet50
from .Dummy import DummyNetwork


def get_model(model_name, num_classes, pretrained=True):
    # TODO add more architectures
    if model_name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'dummy':
        return DummyNetwork(num_classes=num_classes, pretrained=pretrained)
    else:
        raise NotImplementedError

    return model