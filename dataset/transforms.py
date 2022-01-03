import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageChops
# from torchvision.transforms.functional import InterpolationMode # TODO: for torchvision lower than 0.10.1


def get_transform(aug_type):
    cubox_mean = (0.5156, 0.5161, 0.5164)
    cubox_std = (0.1498, 0.1498, 0.1497)
    if aug_type == 'debug':
        train_transform = None
        val_transform = None
    elif aug_type == 'basic':
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cubox_mean, cubox_std)
        ])
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cubox_mean, cubox_std)
        ])
    elif aug_type[:-1] == 'synthetic':
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(224),
            SyntheticPattern(int(aug_type[-1]), (224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cubox_mean, cubox_std)
        ])
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cubox_mean, cubox_std)
        ])
    else:
        raise NotImplementedError

    return train_transform, val_transform


class DummyTransform(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param

    # Not necessary
    def __repr__(self):
        return self.__class__.__name__ + "(param={})".format(self.param)

    def forward(self, img):
        return img


class SyntheticPattern(nn.Module):
    def __init__(self, density, img_size):
        super().__init__()
        self.density = density
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomRotation(60, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomRotation(60, interpolation=Image.BILINEAR),
            transforms.CenterCrop(img_size)
        ])
        tmp = Image.open('./dataset/patterns/{}x.jpg'.format(density))
        self.pattern = np.array(tmp)
        tmp.close()

    def __repr__(self):
        return self.__class__.__name__ + "(densit={},img_size={})".format(self.density, self.img_size)

    def forward(self, img):
        pattern = self.transform(self.pattern)
        return ImageChops.multiply(img, pattern)


class CutMix(nn.Module):
    """
    CutMix: 배치 단위로 적용되며, label 입력을 함께 받아야 함
    """
    def __init__(self, prob=0.5, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
    
    def random_bbox(self, input_size, lam):
        b, _, w, h = input_size
        r_x = np.random.randint(w)
        r_y = np.random.randint(h)
        r_w = np.random.randint(w * np.sqrt(1 - lam))
        r_h = np.random.randint(h * np.sqrt(1 - lam))

        bbx1 = np.clip(r_x - r_w // 2, 0, w)
        bbx2 = np.clip(r_x + r_w // 2, 0, w)
        bby1 = np.clip(r_y - r_h // 2, 0, h)
        bby2 = np.clip(r_y + r_h // 2, 0, h)
        
        return bbx1, bbx2, bby1, bby2

    def __call__(self, pic_tensor, target):
        p = np.random.rand(1)
        # Randomly applied
        if p < self.cutmix_prob:
            return pic_tensor, target, target, 1.0

        lam = np.random.beta(self.alpha, self.alpha)
        b, _, w, h = pic_tensor.shape
        rand_index = torch.randperm(b).cuda()

        target_a = target
        target_b = target[rand_index]

        # Calculate Bounding box for cutting the image
        bbx1, bbx2, bby1, bby2 = self.random_bbox(pic_tensor.shape, lam)

        # Cut & Mix
        pic_tensor[:, :, bbx1:bbx2, bby1:bby2] = pic_tensor[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2-bby1) / (w * h))
        
        # Return mixed image & original label & mixed label & mixed proportion
        return pic_tensor, target_a, target_b, lam


if __name__ == "__main__":
    import os

    src_dir = './dataset/patterns'
    save_dir = './dataset/tmp'
    sample_img = Image.open(os.path.join(save_dir, 'sample.jpg'))
    transform = SyntheticPattern(4, (sample_img.size[1], sample_img.size[0]))
    applied_img = transform(sample_img)
    applied_img.save(os.path.join(save_dir, 'transformed.jpg'))
