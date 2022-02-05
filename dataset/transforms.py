import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageChops, ImageDraw


class WireFenceImg:
    wire_img = None
    threshold = 0

    def set_wire(self, wire_img):
        self.wire_img = wire_img


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
    elif aug_type == 'randsynthetic':
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(224),
            RandomSyntheticPattern((224, 224)),
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
    elif aug_type == "synth_degen":
        train_transform = transforms.Compose([
            UseImage(),
            transforms.ToPILImage(),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cubox_mean, cubox_std)
        ])
        val_transform = transforms.Compose([
            UseImage(),
            transforms.ToPILImage(),
            transforms.CenterCrop(224),
            SingleSyntheticPattern(),
            transforms.ToTensor(),
            transforms.Normalize(cubox_mean, cubox_std)
        ])
    elif aug_type == "box_degen":
        train_transform = transforms.Compose([
            UseImage(),
            transforms.ToPILImage(),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cubox_mean, cubox_std)
        ])
        val_transform = transforms.Compose([
            BoxCalcSyntheticPattern('center'),
            transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cubox_mean, cubox_std)
        ])
    elif aug_type == "pixel_degen":
        train_transform = transforms.Compose([
            UseImage(),
            transforms.ToPILImage(),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cubox_mean, cubox_std)
        ])
        val_transform = transforms.Compose([
            PixelNoise(),
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


class UseImage(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img_params):
        return img_params['img']


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


class RandomSyntheticPattern(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=60, shear=60),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.CenterCrop(img_size)
        ])
        self.patterns = []
        self.p = 0.2
        for density in [1, 2, 4, 8, 9, 10, 11, 12, 13, 16, 17]:
            tmp = Image.open('./dataset/patterns/{}x.jpg'.format(density))
            self.patterns.append(np.array(tmp))
        tmp.close()

    def forward(self, img):
        if self.p < torch.rand(1):
            return img
        else:
            pattern_id = np.random.choice(len(self.patterns))
            pattern = self.transform(self.patterns[pattern_id])
            comb = ImageChops.multiply(img, pattern)
        return comb


class SingleSyntheticPattern(nn.Module):
    def __init__(self):
        super().__init__()
        self.wire = WireFenceImg.wire_img
        
    def forward(self, img):
        # w, h = img.size
        # wire = F.center_crop(self.wire, (h, w))
        wire = self.wire
        comb = ImageChops.multiply(img, wire)
        return comb


class BoxCalcSyntheticPattern(nn.Module):
    def __init__(self, box_loc='center', box_color=(0, 0, 0)):
        super().__init__()
        self.box_loc = box_loc
        self.box_color = box_color
        self.cropper = transforms.CenterCrop(224)

        wire = WireFenceImg.wire_img
        thresh = WireFenceImg.threshold
        wire_thresh = cv2.cvtColor(cv2.threshold(np.array(wire), thresh, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_BGR2GRAY)
        wire_thresh_inv = np.ones_like(wire_thresh)*255 - wire_thresh
        self.wire_inv = Image.fromarray(wire_thresh_inv.astype(np.uint8))

    def calc_area(self, img_metas):
        occl_mask = np.logical_and(self.wire_inv, img_metas['seg_mask']).astype(np.uint8)
        area = occl_mask.sum()
        return area

    def get_box_loc(self, img_metas):
        (xs, ys), (xe, ye) = img_metas['img_locs']
        if self.box_loc == 'center':
            x = (xs + xe) // 2
            y = (ys + ye) // 2

            # r, c = r - (h//2 - 112), c - (r//2 - 112)
            # if r < 0 or c < 0 or r > 112 or c > 112:
            #     r, c = 112, 112
        else:
            raise NotImplementedError(f"Box location {self.box_loc} not implemented")
        return x, y

    def forward(self, img_metas):
        """
        - img_metas
            - img: np array
            - img_locs: 물체 bbox 꼭지점 4곳
        """
        box_area = self.calc_area(img_metas)
        box_r = int(np.sqrt(box_area))

        x, y = self.get_box_loc(img_metas)
        img = Image.fromarray(img_metas['img'].copy())
        img = self.cropper(img)

        draw = ImageDraw.Draw(img)
        draw.rectangle(xy=((x, y), (x+box_r, y+box_r)), fill=tuple(self.box_color))

        return np.array(img)


class PixelNoise(nn.Module):
    def __init__(self, box_loc='center', box_color=(0, 0, 0)):
        super().__init__()
        self.box_loc = box_loc
        self.box_color = box_color
        self.cropper = transforms.CenterCrop(224)

        wire = WireFenceImg.wire_img
        thresh = WireFenceImg.threshold
        wire_thresh = cv2.cvtColor(cv2.threshold(np.array(wire), thresh, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_BGR2GRAY)
        wire_thresh_inv = np.ones_like(wire_thresh)*255 - wire_thresh
        self.wire_inv = Image.fromarray(wire_thresh_inv.astype(np.uint8))

    def calc_area(self, img_metas):
        occl_mask = np.logical_and(self.wire_inv, img_metas['seg_mask']).astype(np.uint8)
        area = occl_mask.sum()
        return area

    def get_pixel_loc(self, img_metas, area):
        nz_cs, nz_rs = (img_metas['seg_mask']).nonzero()
        inds = np.random.choice(range(len(nz_cs)), area, replace=False)
        
        if len(inds) == 0:
            return nz_cs, nz_rs

        return nz_cs[inds], nz_rs[inds]

    def forward(self, img_metas):
        """
        - img_metas
            - img: np array
            - img_locs: 물체 bbox 꼭지점 4곳
        """
        occ_area = self.calc_area(img_metas)
        pixel_cs, pixel_rs = self.get_pixel_loc(img_metas, occ_area)

        img = img_metas['img'].copy()
        img = Image.fromarray(img)
        img = np.array(self.cropper(img))

        for pixel_c, pixel_r in zip(pixel_cs, pixel_rs):
            img[pixel_c, pixel_r, :] = 0
        img = Image.fromarray(img)

        return np.array(img)


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
