import torch
import torch.utils.data
import torch.nn.parallel
from datetime import datetime
import pytz

import os
import tqdm
import torch
import models
import argparse

from configs import data_config
from dataset import get_test_loader
from utils.eval_meter import Evaluate
from utils import print_log, get_root_logger

KST = pytz.timezone('Asia/Seoul')

def simple_inference(test_loader, model, evaluator, logger):
    """
    Run evaluation
    """
    # print("Start inference...", datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S,%z"))
    print_log("Start inference...", logger=logger)
    model.eval()
    # print(val_loader.dataset[0])

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_loader)):
            img = batch['img'].cuda()
            # compute output
            output = torch.softmax(model(img), dim=1)
            # measure accuracy and record loss
            evaluator.update(output, batch)

    return evaluator


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cubox', type=str, choices=['cubox'])
parser.add_argument('--data_root', default='/mnt/disk1/cubox_dataset/original/images', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--data_config', type=str)
parser.add_argument('--transform_type', default='basic')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--not_pretrain', action='store_false')
parser.add_argument('--optim', default='sgd', choices=['sgd', 'adam', 'adamw'])
parser.add_argument('--sched', default='cos', choices=['multi', 'non', 'cos', 'cos_warm'])
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--ckpt_dir', type=str, required=True)
parser.add_argument('--ckpt_file', type=str, default='epoch80.pth')

"""
Example: CUDA_VISIBLE_DEVICES=4 python main.py --method=finetune --data_config=none2all --transform_type=basic --epochs=100 
"""

# TODO mode for only evaluation of already trained model
if __name__ == "__main__":
    # os.environ['TZ'] = 'Asia/Seoul'
    # time.tzset()
    logger = get_root_logger()
    print_log("Evaluate!", logger=logger)

    args = parser.parse_args()
    data_conf = data_config[args.dataset][args.data_config]
    trained_args = torch.load(os.path.join(args.ckpt_dir, 'args.pth'))

    test_loader, classes = get_test_loader(data_conf, data_root=args.data_root,
                                                transform_type=args.transform_type, args=args)
    # TODO refactor
    model = models.get_model(trained_args.arch, num_classes=len(classes), pretrained=trained_args.not_pretrain)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, args.ckpt_file))['state_dict'])
    model.cuda()

    evaluator = Evaluate(classes)
    evaluator = simple_inference(test_loader, model, evaluator, logger)
    print_log("Inference Done!", logger=logger)
    # evaluate on test set
    # print("Start evaluating with inferenced results... ", datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S,%z"))
    print_log("Start evaluating with inferenced results... ", logger=logger)
    evaluator.summarize()

    # print("Evaluation end!", datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S,%z"))
    print_log("Evaluation end!", logger=logger)
