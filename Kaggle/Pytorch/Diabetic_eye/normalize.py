"""
Script need to normalize and preprocessing images.
Using trim method (for cut black zones at images)
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2
import warnings
import numpy as np
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm


# Data augmentation for images
train_transforms = A.Compose(
    [
        A.Resize(width=760, height=760),
        A.RandomCrop(height=728, width=728),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(p=0.3),
        A.CLAHE(p=0.3),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
        A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
        A.Normalize(
            mean=[0.3199, 0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=728, width=728),
        A.Normalize(
            mean=[0.3199, 0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()

def trim(image):
    img = np.array(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = img_gray > 0.1 * np.mean(img_gray[img_gray != 0])
    row_sums = np.sum(im, axis=1)
    col_sums = np.sum(im, axis=0)   
    rows = np.where(row_sums > img.shape[1] * 0.02)[0]
    cols = np.where(col_sums > img.shape[0] * 0.02)[0]
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    cropped_image = img[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(cropped_image)

def resize_maintain_aspect(image, desired_size):
    old_size = image.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = image.resize(new_size, Image.ANTIALIAS)
    new_img_size = Image.new("RGB", (desired_size, desired_size))
    new_img_size.paste(img, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_img_size

def fast_image_resize(input_path_folder, output_path_folder, output_size=None):
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=100")
        exit()

    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)

    jobs = [
        (file, input_path_folder, output_path_folder, output_size)
        for file in os.listdir(input_path_folder)
    ]

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))

def save_single(args):
    img_file, input_path_folder, output_path_folder, output_size = args
    image_original = Image.open(os.path.join(input_path_folder, img_file))
    image = trim(image_original)
    image = resize_maintain_aspect(image, desired_size=output_size[0])
    image.save(os.path.join(output_path_folder + img_file))

