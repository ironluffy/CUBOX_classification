import os
import tqdm
import time
import torch
import models
import numpy as np
import importlib
import argparse

from configs import data_config
from dataset import get_loaders
from utils import save_checkpoint, get_optimizer, get_lr_scheduler, get_root_logger, print_log
from classification.eval import validate
from utils.logging import get_root_logger


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cubox', type=str, choices=['cubox'])
parser.add_argument('--experiment', required=True, type=str)
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

"""
Example: CUDA_VISIBLE_DEVICES=4 python main.py --method=finetune --data_config=none2all --transform_type=basic --epochs=100 
"""

# TODO mode for only evaluation of already trained model
if __name__ == "__main__":

    args = parser.parse_args()
    data_conf = data_config[args.dataset][args.data_config]
    args.save_dir = os.path.join('./classification_checkpoints', args.method, args.data_config, args.transform_type, args.experiment, 
                                 time.strftime("%Y%m%d-%H%M%S"))

    logger = get_root_logger()

    # cudnn.benchmark = True
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    torch.save(args, os.path.join(args.save_dir, 'args.pth'))

    train_loader, val_loaders, test_loaders, num_classes = get_loaders(args.dataset, data_conf, data_root=args.data_root,
                                                                       transform_type=args.transform_type, args=args)

    model = models.get_model(args.arch, num_classes=num_classes, pretrained=args.not_pretrain)
    model.cuda()
    optimizer = get_optimizer(args.dataset, args.optim, model, args.lr)
    lr_scheduler = get_lr_scheduler(args.dataset, args.sched, optimizer, args.epochs)

    # TODO if there are more tasks, need to be generalized
    # import train function from given method
    method = importlib.import_module('classification.{}'.format(args.method))
    train = method.train

    save_checkpoint({
        'epoch': 0,
        'state_dict': model.state_dict(),
    }, False, filename=os.path.join(args.save_dir, 'epoch0.pth'))

    # define loss function (criterion) and optimizer

    best_val = 0
    best_val_test = 0
    is_best = False

    epoch_storage = method.init_storage()
    for epoch in tqdm.trange(args.epochs):

        # train for one epoch
        train(train_loader, model, optimizer, epoch + 1, args, epoch_storage, logger=logger)
        if lr_scheduler is not None:
            lr_scheduler.step()

        # evaluate on validation set

        val_obs, test_obs = dict(), dict()
        for i, val_loader in enumerate(val_loaders):
            val_obs[data_conf["val"][i]] = validate(val_loader, model, prefix='val/{}'.format(data_conf["val"][i]),
                                                    epoch=epoch + 1, logger=logger)
        # evaluate on test set
        for i, test_loader in enumerate(test_loaders):
            test_obs[data_conf["val"][i]] = validate(test_loader, model, prefix='test/{}'.format(data_conf["test"][i]),
                                                     epoch=epoch + 1, logger=logger)

        val_accs = []
        for occl in data_conf["train"]:
            val_accs.append(val_obs[occl].accuracy.avg.cpu().numpy())
        val_acc = np.mean(val_accs)

        test_accs = []
        for occl in data_conf["train"]:
            test_accs.append(test_obs[occl].accuracy.avg.cpu().numpy())
        test_acc = np.mean(test_accs)

        # remember best prec@1 and save checkpoint
        if val_acc > best_val:
            best_val_test = test_accs
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'best_val_test': best_val_test,
            }, is_best=True, filename=os.path.join(args.save_dir, 'epoch{}.pth'.format(epoch)))
        best_val = max(val_acc, best_val)

        if epoch > 0 and epoch % 20 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
            }, is_best, filename=os.path.join(args.save_dir, 'epoch{}.pth'.format(epoch)))

    for i, occl in enumerate(data_conf["train"]):
        print_log({"{}/Best Val. Test Acc.".format(occl): test_accs[i]}, logger=logger)
