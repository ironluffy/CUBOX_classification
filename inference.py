import torch
import models
import argparse
import torch.nn as nn

from configs import data_config
from dataset import get_test_loader
from classification.eval import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cubox', type=str, choices=['cubox'])
parser.add_argument('--data_root', default='./wired_v0', type=str)
parser.add_argument('--data_config', type=str, default='none2none')
parser.add_argument('--transform_type', default='basic')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--ckpt_dir', type=str)
parser.add_argument('--ckpt_name', type=str)


def inference(args):
    # Get test loader
    train_args = torch.load(f"{args.ckpt_dir}/args.pth")
    data_conf = data_config[args.dataset][args.data_config]
    test_loader, classes = get_test_loader(data_conf, args.data_root, args.transform_type, args)

    # Import model
    model = models.get_model(train_args.arch, num_classes=len(classes), pretrained=False, dist=False)
    ckpt = torch.load(f"{args.ckpt_dir}/{args.ckpt_name}")['state_dict']
    ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}
    model.load_state_dict(ckpt)
    model.cuda()
    model.eval()

    # Evaluate on test set
    test_obs = evaluate(test_loader, model, prefix='', epoch=0)
    print(f"Checkpoint: {args.ckpt_dir}/{args.ckpt_name}")
    print(f"Model: {train_args.arch}+{train_args.data_config}+{getattr(train_args, 'mixed_transform', None)}+{train_args.transform_type}")
    print(f"Dataset: {args.data_root}+ {args.data_config}")
    print("Result: ")
    print(test_obs.get_info_dict())


if __name__ == "__main__":
    args = parser.parse_args()
    inference(args)