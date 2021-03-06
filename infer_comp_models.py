import torch
import torch.utils.data
import torch.nn.parallel
import pytz

import os
import tqdm
import torch
from torch.utils.data import DataLoader
import models
import argparse
import torchvision.transforms as transforms
from PIL import Image

from configs import data_config, wire_cfg, transform_config
from dataset import get_transform, get_test_dataset, WireFenceImg
# from synthwire_img import wire_cfg


def create_wire(wire_dict, save_dir, logger):
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
    wandb.log({'wire_img': wandb.Image(wire)})

    print_log(f"Synthetic wire saved to path {save_path}", logger=logger)
    print_log(wire_dict, logger=logger)
    wandb.log({'wire_config': wire_dict})

    return wire


def save_denorm_sample(img, img_name, save_path):
    img_name = img_name.split("/")[-1]
    cubox_mean = torch.Tensor((0.5156, 0.5161, 0.5164))
    cubox_std = torch.Tensor((0.1498, 0.1498, 0.1497))
    unnormalize = transforms.Normalize((-cubox_mean / cubox_std).tolist(), (1.0 / cubox_std).tolist())

    # img = img.mul_(cubox_std).add_(cubox_mean)
    img = unnormalize(img)
    img = transforms.ToPILImage()(img)
    img.save(f"{save_path}/{img_name}")

    # save_path.split("/")[-1] # transform type
    # img_name.split("_")[0] # class name
    wandb.log({f'{save_path.split("/")[-1]}/{img_name.split("_")[0]}': wandb.Image(img)})


def simple_inference(test_loader, model, evaluator, model_name='', transform_type='', save_path=None, logger=None):
    """
    Run evaluation
    """
    if not logger:
        logger = get_root_logger()

    print_log(f"Start inference {model_name.capitalize()}-{transform_type}...", logger)
    model.eval()
    evaluator.reset()

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
    'wire_no': '17',
    'affine_degree': 60,
    'shear_coords': (10, 5),
    'distort_scale': 0.5,
    'perspective': True,
}

TRANSFORM_TYPES = ['synth_obj_region', 'synth_bgr_region', 'synth_degen', 'box_degen', 'pixel_degen']
# TRANSFORM_TYPES = ['basic']
# TRANSFORM_TYPES = ['synth_obj_region', 'synth_bgr_region', 'synth_degen', 'box_degen', 'pixel_degen', 'box_white', 'box_wire']

# TODO: transform_dict ???????????? wandb ???????????? transforms??? ????????? ????????????

SAVE_SAMPLES = True


# TODO mode for only evaluation of already trained model
if __name__ == "__main__":
    import wandb
    import time
    from utils.eval_meter import Evaluate
    from utils.logging import get_root_logger, print_log

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--dataset', type=str, default='cubox_singlewire')
    parser.add_argument('--made_wire', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)

    parser.add_argument('--data_config', type=str, default="none2none")
    parser.add_argument('--transform_type', default='synth_degen') # Apply synthetic augmentation
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--result_dir', default='./pred_result')

    args = parser.parse_args()
 
    start_time = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join('./synth_results', (start_time))
    os.makedirs(save_dir)
    logger = get_root_logger(log_file=f"{save_dir}/result.txt")
    print_log("Evaluate!", logger)

    wandb.init(project="OBWF_results", entity="noisy-label")
    if args.run_name: 
        wandb.run.name = args.run_name
    else: 
        wandb.run.name = f"{start_time}/{WIRE_DICT['wire_no']}"

    # Make synthetic wire image
    if args.made_wire:
        synth_wire = Image.open(args.made_wire)
    else:
        synth_wire = create_wire(WIRE_DICT, save_dir, logger)
    WireFenceImg.wire_img = synth_wire
    WireFenceImg.threshold = wire_cfg[WIRE_DICT['wire_no']]['threshold']
    
    data_conf = data_config[args.dataset][args.data_config]
    
    test_dataset, classes = get_test_dataset(data_conf, data_root=args.data_root, transform_type=args.transform_type)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    transform_types = TRANSFORM_TYPES

    save_samples = SAVE_SAMPLES
    acc_table = wandb.Table(columns=["run_name", "model_name", "transform_type", "acc", "top5"])
    # aug_table = wandb.Table(columns=["run_name", "transform_type", "cls_name", "img", "img_name"])
    for model_name, best_ckpt in BEST_CKPTS.items():
        ckpt_dir = os.path.dirname(best_ckpt)
        trained_args = torch.load(os.path.join(ckpt_dir, 'args.pth'))
        model = models.get_model(trained_args.arch, num_classes=len(classes), pretrained=trained_args.not_pretrain)
        model.load_state_dict(torch.load(best_ckpt)['state_dict'])
        model.cuda()
        
        for transform_type in transform_types:
            evaluator = Evaluate(classes, logger)
            _, val_transform = get_transform(transform_type)
            test_dataset.transform = val_transform
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)

            if save_samples:            
                os.makedirs(f"{save_dir}/{transform_type}")
                evaluator = simple_inference(test_loader, model, evaluator, model_name, transform_type=transform_type, save_path=f"{save_dir}/{transform_type}", logger=logger)
            else:
                evaluator = simple_inference(test_loader, model, evaluator, model_name, transform_type=transform_type, save_path=None, logger=logger)

            acc_dict = evaluator.summarize()
            acc_table.add_data(wandb.run.name, model_name, transform_type, acc_dict['top1'], acc_dict['top5'])
        save_samples = False
        model.cpu()
        del(model)

    wandb.log({"Accuracy Metric": acc_table})
    print_log("Inference Done!", logger)