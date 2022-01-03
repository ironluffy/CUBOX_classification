import os
import cv2
import tqdm
import shutil
import argparse
from multiprocessing import Process


def createFolder(src_path: str, dst_path: str) -> None:
    """
    :param src_path:
    :param dst_path:
    :return:
    """
    dir_list = os.listdir(src_path)
    for directory in dir_list:
        directory = directory.capitalize()
        try:
            if not os.path.exists(os.path.join(dst_path, directory)):
                os.makedirs(os.path.join(dst_path, directory))
        except OSError:
            print('Error: Creating directory. ' + directory)


def get_filename_per_cls(path):
    classes = os.listdir(path)
    file_list = []
    no_jpg_file_list = []

    for cls in tqdm.tqdm(classes):
        cls_path = os.path.join(path, cls)
        for filename in os.listdir(cls_path):
            full_filename = os.path.join(cls_path, filename)
            file_extension = full_filename.split(".")[-1]
            if file_extension == "JPG" or file_extension == "jpeg" or file_extension == "jpg":
                file_list.append(full_filename)
            else:
                print(file_extension)
                no_jpg_file_list.append(full_filename)

    return file_list, no_jpg_file_list


def copy_all_file(file_list, new_path):
    # TODO: Think about resizing and copy all file at the same time
    for src_path in tqdm.tqdm(file_list):
        file = src_path.split("/")[-1]
        file_split = file.split("_")
        cls_name = file_split[0].capitalize()
        if cls_name == "Hairbrush":
            cls_name = "Hairbrushcomb"
        elif cls_name == "Frenchloaf":  # Loaf 폴더에 Frenchloaf로 되어있었음
            cls_name = "Loaf"
        elif cls_name == "Coffeemug":
            cls_name = "Mug"
        elif cls_name == "Stepler":  # Stapler 폴더에 stepler라고 파일이 들어있었음
            cls_name = "Stapler"
        elif cls_name == "Electric":
            cls_name = "Electricswitch"
            del file_split[1]
        file_split[0] = cls_name
        file = "_".join(file_split)
        shutil.copyfile(src_path, new_path + "/" + cls_name + "/" + file)


def resize_img_save(image_origin):
    if image_origin.shape[1] >= image_origin.shape[0]:
        ratio = image_origin.shape[0] / 300
        length_another = round(image_origin.shape[1] / ratio)
        image_resize = cv2.resize(image_origin, (length_another, 300))
    else:
        ratio = image_origin.shape[1] / 300
        length_another = round(image_origin.shape[0] / ratio)
        image_resize = cv2.resize(image_origin, (300, length_another))
    return image_resize


def resize_and_move(src_path):
    filenames = os.listdir(src_path)
    for filename in tqdm.tqdm(filenames):
        full_path = os.path.join(src_path, filename)
        if os.path.isdir(full_path):
            continue
        else:
            occlusion_name = filename.split("_")[-1].split(".")[0]
            occlusion_path = os.path.join(src_path, occlusion_name)
            if not os.path.exists(occlusion_path):
                os.makedirs(occlusion_path)
            try:
                org_img = cv2.imread(full_path, cv2.IMREAD_COLOR)
                resized = resize_img_save(org_img)
                cv2.imwrite(os.path.join(occlusion_path, filename), resized)
            except:
                continue


def split_resize_occlusion(src_path):
    classes = sorted(os.listdir(src_path))
    proc_list = []
    for cls in classes:
        cls_path = os.path.join(src_path, cls)
        proc = Process(target=resize_and_move, args=(cls_path,))
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()


parser = argparse.ArgumentParser(description='CUBOX dataset preprocess')
parser.add_argument('--dst_path', type=str, default='../../datasets/cubox')
parser.add_argument('--src_path', type=str, default='../../datasets/images_v2')

if __name__ == "__main__":
    args = parser.parse_args()
    # createFolder(args.src_path, args.dst_path)
    # file_list, no_jpg_file_list = get_filename_per_cls(args.src_path)
    # copy_all_file(file_list, args.dst_path)
    split_resize_occlusion(args.dst_path)
