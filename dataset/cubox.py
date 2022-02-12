import os
import re
import numpy as np
from .transforms import get_transform
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

occlusion_types = ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"]


class CUBOXdataset(Dataset):
    def __init__(self, root: str, occlusion: str, split: str, transform=None):
        self.split = split
        self.occlusion = occlusion
        # self.dataset_path = os.path.join(root, 'cubox', self.split, self.occlusion)
        self.dataset_path = os.path.join(root, self.split, self.occlusion)
        self.classes = sorted(os.listdir(self.dataset_path))
        self.transform = transform
        self.imgs = []
        self.img_names = []
        self.labels = []
        self.selected_sample = []

        for cls_id, cls in enumerate(self.classes):
            cls_path = os.path.join(self.dataset_path, cls)
            is_sample = True
            for filename in sorted(os.listdir(cls_path)):
                self.img_names.append(os.path.join(cls_path, filename))
                self.selected_sample.append(is_sample)
                if is_sample: is_sample = False
                self.labels.append(cls_id)

        self.num_classes = len(self.classes)

    def __len__(self):
        # return len(self.imgs)
        return len(self.img_names)

    def calc_img_loc(self, img_name):
        seg_name = re.sub('.jpeg', '.png', re.sub('.jpg', '.png', re.sub('images', 'seg_map', img_name)))
        try:
            seg_map = Image.open(seg_name)
        except:
            print(img_name, seg_name)
            return None, None
        
        seg_map = transforms.CenterCrop(224)(seg_map)
        seg_map = np.array(seg_map)
        nz_cs, nz_rs = (seg_map).nonzero()

        try:
            area_range = ((min(nz_rs), min(nz_cs)), (max(nz_rs), max(nz_cs))) # (rs, cs), (re, ce)
        except:
            print(seg_name)
            return seg_map, ((0, 0), (0, 0))

        return seg_map, area_range


    def __getitem__(self, idx):
        item_dict = dict()
        # img = self.imgs[idx]
        img_name = self.img_names[idx]
        img = Image.open(img_name)
        img = np.array(img)

        seg_mask, img_locs = self.calc_img_loc(img_name)

        img_metas = {
            'img_name': img_name,
            'img': img,
            'img_locs': img_locs,
            'seg_mask': seg_mask,
        }

        if self.transform is not None:
            img = self.transform(img_metas)
        item_dict['img'] = img
        label = self.labels[idx]
        item_dict['gt'] = label
        item_dict['img_name'] = img_name
        item_dict['is_sample'] = self.selected_sample[idx]
        return item_dict


if __name__ == "__main__":
    root = '../datasets'
    train_transform, val_transform = get_transform('basic')
    trainset = CUBOXdataset(root, split='train', occlusion='none', transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    batch = iter(trainloader).next()
    print(batch['label'])
