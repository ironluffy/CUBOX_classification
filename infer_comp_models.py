from cv2 import transform
import torch
import torch.utils.data
import torch.nn.parallel
from datetime import datetime
import pytz

import os
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
import models
import argparse
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageDraw

from configs import data_config
from dataset import get_test_loader, get_transform, WireFenceImg, get_test_dataset
from synthwire_img import wire_cfg


KST = pytz.timezone('Asia/Seoul')


def create_wire(wire_dict, save_dir):
    wire_no = wire_dict['wire_no']
    affine_degree = wire_dict['affine_degree']
    shear_coords = wire_dict['shear_coords']
    distort_scale = wire_dict['distort_scale']
    perspective = wire_dict['perspective']
    
    p = int(perspective)
    shear_x, shear_y = shear_coords

    wire_transform = transforms.Compose([
        transforms.RandomAffine(degrees=[affine_degree, affine_degree], shear=(shear_x,shear_x,shear_y,shear_y), fill=255),
        transforms.RandomPerspective(distortion_scale=distort_scale, p=p, fill=255),
        transforms.CenterCrop(224),
    ])

    wire = Image.open(wire_cfg[wire_no]['wire_path'])
    wire = wire_transform(wire)


    save_path = f"{save_dir}/synth_wire.png"
    wire.save(save_path)
    print(f"Synthetic wire saved to path {save_path}", datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S,%z"))

    return wire


def save_denorm_sample(img, img_name, save_path):
    img_name = img_name.split("/")[-1]
    cubox_mean = torch.Tensor((0.5156, 0.5161, 0.5164))
    cubox_std = torch.Tensor((0.1498, 0.1498, 0.1497))
    unnormalize = transforms.Normalize((-cubox_mean / cubox_std).tolist(), (1.0 / cubox_std).tolist())

    # img = img.mul_(cubox_std).add_(cubox_mean)
    img = unnormalize(img)
    transforms.ToPILImage()(img).save(f"{save_path}/{img_name}")


def simple_inference(test_loader, model, evaluator, model_name='', transform_type='', save_path=None):
    """
    Run evaluation
    """
    print(f"Start inference {model_name.capitalize()}-{transform_type}...", datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S,%z"))
    model.eval()

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_loader)):
            img = batch['img'].cuda()

            # compute output
            output = torch.softmax(model(img), dim=1)
            
            # Save augmented sample
            if save_path:
                sample_ids = batch['is_sample'].nonzero()
                for j in sample_ids.squeeze():
                    save_denorm_sample(img[j], batch['img_name'][j], save_path)

            # measure accuracy and record loss
            evaluator.update(output, batch)

    return evaluator
"""
Example: CUDA_VISIBLE_DEVICES=4 python main.py --method=finetune --data_config=none2all --transform_type=basic --epochs=100 
"""

BEST_CKPTS = {
    'NONE_BEST_VAL' : "/home/yura/Computer_Vision_LAB/CUBOX/cubox_classification_from_kdm/saved_results/finetune/none2all/basic/20220127-114452/epoch40.pth",
    'SYNTH_BEST_VAL' : "/home/yura/Computer_Vision_LAB/CUBOX/cubox_classification_from_kdm/saved_results/finetune/synth2all/randsynthetic/20220127-113800/epoch32.pth",
    'ALL_BEST_VAL' : "/home/yura/Computer_Vision_LAB/CUBOX/cubox_classification_from_kdm/saved_results/finetune/all2all/basic/20220127-090402/epoch20.pth",
}

WIRE_DICT = {
    'wire_no': '9',
    'affine_degree': 60,
    'shear_coords': (30, 45),
    'distort_scale': 0.5,
    'perspective': True,
}

# TODO mode for only evaluation of already trained model
if __name__ == "__main__":
    import re
    import time
    from utils.eval_meter import Evaluate

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--dataset', type=str, default='cubox_singlewire')

    parser.add_argument('--data_config', type=str)
    parser.add_argument('--transform_type', default='synth_degen') # Apply synthetic augmentation
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--result_dir', default='./pred_result')

    print("Evaluate!", datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S,%z"))
    save_dir = os.path.join('./synth_results', (time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(save_dir)

    args = parser.parse_args()

    # Make synthetic wire image
    synth_wire = create_wire(WIRE_DICT, save_dir)
    WireFenceImg.wire_img = synth_wire
    WireFenceImg.threshold = wire_cfg[WIRE_DICT['wire_no']]['threshold']
    
    data_conf = data_config[args.dataset][args.data_config]
    
    test_dataset, classes = get_test_dataset(data_conf, data_root=args.data_root, transform_type=args.transform_type)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    transform_types = ['synth_degen', 'box_degen']

    save_samples = True
    for model_name, best_ckpt in BEST_CKPTS.items():
        ckpt_dir = os.path.dirname(best_ckpt)
        trained_args = torch.load(os.path.join(ckpt_dir, 'args.pth'))
        model = models.get_model(trained_args.arch, num_classes=len(classes), pretrained=trained_args.not_pretrain)
        model.load_state_dict(torch.load(best_ckpt)['state_dict'])
        model.cuda()
        
        evaluator = Evaluate(classes)
        for transform_type in transform_types:
            _, val_transform = get_transform(transform_type)
            test_dataset.transform = val_transform
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)

            if save_samples:            
                os.makedirs(f"{save_dir}/{transform_type}")
                evaluator = simple_inference(test_loader, model, evaluator, model_name, transform_type=transform_type, save_path=f"{save_dir}/{transform_type}")
            else:
                evaluator = simple_inference(test_loader, model, evaluator, model_name, transform_type=transform_type, save_path=None)

            evaluator.summarize()
        
        save_samples = False
        model.cpu()
        del(model)

    print("Inference Done!", datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S,%z"))