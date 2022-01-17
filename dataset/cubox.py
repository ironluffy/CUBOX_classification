import os
import numpy as np
from .transforms import get_transform
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

        for cls_id, cls in enumerate(self.classes):
            cls_path = os.path.join(self.dataset_path, cls)
            for filename in os.listdir(cls_path):
                # img = Image.open(os.path.join(cls_path, filename))
                self.img_names.append(os.path.join(cls_path, filename))
                # self.imgs.append(np.array(img))
                # img.close()
                self.labels.append(cls_id)

        self.num_classes = len(self.classes)

    def __len__(self):
        # return len(self.imgs)
        return len(self.img_names)

    def __getitem__(self, idx):
        item_dict = dict()
        # img = self.imgs[idx]
        img_name = self.img_names[idx]
        if self.transform is not None:
            img = Image.open(img_name)
            img = np.array(img)
            img = self.transform(img)
        item_dict['img'] = img
        label = self.labels[idx]
        item_dict['gt'] = label
        item_dict['img_name'] = img_name
        return item_dict


if __name__ == "__main__":
    root = '../datasets'
    train_transform, val_transform = get_transform('basic')
    trainset = CUBOXdataset(root, split='train', occlusion='none', transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    batch = iter(trainloader).next()
    print(batch['label'])
