import os
import tqdm
import argparse
from dataset import get_test_loader
from configs import data_config
import torchvision.transforms as transforms


# Test set configs
DATA_TYPE = 'cubox'
DATA_CONF = 'none2none'
DATA_ROOT = "/mnt/disk1/cubox_dataset/original/images"
TEST_TRANSFORM = "wired_v0"

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

# Make save dir
save_dir = f"./{TEST_TRANSFORM}"
os.makedirs(save_dir, exist_ok=True)
data_conf = data_config[DATA_TYPE][DATA_CONF]


# Create dataset!
test_loader, classes = get_test_loader(data_config=data_conf, data_root=DATA_ROOT, transform_type=TEST_TRANSFORM, args=args)
to_pil = transforms.ToPILImage()
for i, batch in tqdm.tqdm(enumerate(test_loader)):
    for img, img_name in zip(batch['img'], batch['img_name']):
        save_path = f"{save_dir}/{'/'.join(img_name.split('/')[-4:])}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        to_pil(img).save(save_path)
