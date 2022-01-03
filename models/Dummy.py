import torch
import torch.nn as nn


class DummyNetwork(nn.Module):
    def __init__(self, num_classes: int, in_dim: int = 10, hid_dim: int = 10, pretrained: bool = False):
        super(DummyNetwork, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, num_classes)
        if pretrained:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> dict:
        """
        :param x: Input tensor (image Tensor)
        :return: Dictionary of output and byproducts (optional)
        """
        out_dict = dict()
        feat = self.linear1(x)
        out = self.linear2(feat)

        out_dict['feat'] = feat
        out_dict['out'] = out

        return out_dict


if __name__ == "__main__":
    net = DummyNetwork(4, 5, 3)
    rand_in = torch.randn((3, 4))
    out = net(rand_in)
