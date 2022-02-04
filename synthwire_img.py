from PIL import Image, ImageChops, ImageDraw
import cv2
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
import os

from transformers import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP


def synth_aug(save_dir, wire_path, orig_img, degree=60):
    """
    - Add synthetic pattern to unoccludede iamge
    - Input: 
        - save_dir: path to save synthesized data
        - wire_path: path to occlusion
        - orig_img: PIL original iamge
        - degree: transform argument! how much to rotate wire
    - Output: 
        - comb: wired image
        - wire: transformed wire
    """
    w, h = orig_img.size
    # make path
    os.makedirs(save_dir, exist_ok=True)

    # TODO: Random perspective 대신에, 지정된 형식으로 철망을 가릴 수 있도록 하기
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=degree, shear=60, fill=255),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=255),
        transforms.CenterCrop((h, w))
    ])    
    wire = Image.open(wire_path)
    wire = transform(wire)

    comb = ImageChops.multiply(orig_img, wire)
    comb.save(f"{save_dir}/wire_synth.jpg")
    return comb, wire


def calc_synth_cross_obj(wire_cfg, wire_img, seg_map, save_dir=None):
    """
    Calculate occluded area of the image
    - Inputs
        - wire_cfg: wire config containing binary threshold
        - wire_img: transformed wire image
        - seg_map: ground truth segmentation map
        - save_dir: path to save wire & occluded region mask
    """
    thresh = wire_cfg['threshold']
    seg_img = np.array(seg_map)
    obj_mask = (seg_img > 0).astype(int)

    # wire_thresh shape: (w, h)
    wire_thresh = cv2.cvtColor(cv2.threshold(np.array(wire_img), thresh, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_BGR2GRAY)
    wire_thresh_inv = np.ones_like(wire_thresh)*255 - wire_thresh

    occl_mask = np.logical_and(wire_thresh_inv, obj_mask).astype(np.uint8)
    if save_dir:
        Image.fromarray(occl_mask * 255).save(f"{save_dir}/occluded_region.png")
        Image.fromarray(wire_thresh_inv).save(f"{save_dir}/wire_region.png")

    area = occl_mask.sum()

    return area


def create_box_mask(orig_img, seg_map, save_dir, box_size, box_color=(0, 0, 0)):
    """
    - Augment data with same-occlusion sized box & save box-masked image
    - Box color can be changed as the mean RGB value of the none occluded image
    """
    box_r = int(np.sqrt(box_size))

    nz_cs, nz_rs = (np.array(seg_map)).nonzero()
    area_range = ((min(nz_rs), min(nz_cs)), (max(nz_rs), max(nz_cs)))
    (rs, cs), (re, ce) = area_range


    for r in range(rs, max(rs+1, re - box_r)):
        for c in range(cs, max(cs+1, ce - box_r)):
            temp_img = orig_img.copy()
            draw = ImageDraw.Draw(temp_img)
            draw.rectangle(xy=((r, c), (r+box_r, c+box_r)), fill=tuple(box_color))
            temp_img.save(f"{save_dir}/{r}_{c}.png")


wire_cfg = {
    '1': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/1x.jpg',
        'threshold': 100,
    },
    '2': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/2x.jpg',
        'threshold': 100,
    },
    '3': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/3x.jpg',
        'threshold': 100,
    },
    '4': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/4x.jpg',
        'threshold': 100,
    },
    '8': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/8x.jpg',
        'threshold': 155,
    },
    '9': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/9x.jpg',
        'threshold': 155,
    },
    '10': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/10x.jpg',
        'threshold': 210, # bad threshold..
    },
    '11': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/11x.jpg',
        'threshold': 125,
    },
    '13': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/13x.jpg',
        'threshold': 125,
    },
    '16': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/16x.jpg',
        'threshold': 125,
    },
    '17': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/17x.jpg',
        'threshold': 125,
    },
    '100': {
        'wire_path': None,
        'threshold': None,
    },
}


if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--wire_no", type=str, choices=list(wire_cfg.keys()))
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="/mnt/disk1/cubox_dataset/pattern_synth")

    args = parser.parse_args()

    img_path = args.img_path
    orig_img = Image.open(img_path)
    seg_path = re.sub("images", "seg_map", re.sub(".jpg", ".png", img_path))
    seg_map = Image.open(seg_path)
    wire_no_cfg = wire_cfg[args.wire_no]
    
    img_name = Path(img_path).stem 
    wire_name = Path(wire_no_cfg['wire_path']).stem 
    save_dir = f"{args.save_dir}/{img_name}_{args.wire_no}"


    comb_img, wire_img = synth_aug(save_dir, wire_no_cfg['wire_path'], orig_img, degree=60)
    synth_over_img_area = calc_synth_cross_obj(wire_no_cfg, wire_img, seg_map, save_dir)

    create_box_mask(orig_img, seg_map, save_dir, synth_over_img_area, box_color=[0, 0, 0])
