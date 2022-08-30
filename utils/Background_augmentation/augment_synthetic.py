import cv2
import json
import os
import numpy as np
import albumentations as A
from joblib import Parallel, delayed
from pathlib import Path
from random import sample, uniform
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import hydra
from omegaconf import DictConfig, OmegaConf

_IMG_EXT = [".jpg", ".png", ".jpeg"]


def load_scene(scene_path, add_mask_prefix=""):
    json_dir, json_fname = os.path.split(scene_path)
    json_noext, _ = os.path.splitext(json_fname)
    img_path = os.path.join(json_dir, json_noext + ".png")

    img = cv2.imread(os.path.join(json_dir, img_path))
    with open(scene_path) as f:
        scene_dict = json.load(f)
    mask_to_path = {}
    for obj in scene_dict["objects"]:
        visible_mask = cv2.imread(os.path.join(json_dir, obj["visible_mask"]))
        full_mask = cv2.imread(os.path.join(json_dir, obj["full_mask"]))

        obj["visible_mask"] = add_mask_prefix + obj["visible_mask"]
        obj["full_mask"] = add_mask_prefix + obj["full_mask"]

        mask_to_path[obj["visible_mask"]] = visible_mask
        mask_to_path[obj["full_mask"]] = full_mask

    return scene_dict, img, mask_to_path


def create_masks_dir(mask_path):
    mask_dir, _ = os.path.split(mask_path)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)


def correct_and_save_scene(dest_dir, scene_dict, img, path_to_mask, used_foreground):
    """Saves the scene with all available valid information

    Args:
        dest_dir (str): Path to save scene(
        scene_dict (dict): Dictionary that contains scene description
        img (array): Scene image
        path_to_mask (dict): Dictionary that contains map from $scene_dict path to real path orig masks
        used_foreground (bool): Flag that marks translation is not valid
    """
    scene_dict.pop("scene_depth")
    for i, obj in enumerate(scene_dict["objects"]):
        obj["id"] = i
        if used_foreground:
            # Is not valid
            obj.pop("translation")
        visible_mask = os.path.join(dest_dir, obj["visible_mask"])
        create_masks_dir(visible_mask)
        cv2.imwrite(visible_mask, path_to_mask[obj["visible_mask"]])
        full_mask = os.path.join(dest_dir, obj["full_mask"])
        create_masks_dir(full_mask)
        cv2.imwrite(full_mask, path_to_mask[obj["full_mask"]])

    cv2.imwrite(os.path.join(dest_dir, scene_dict["scene"]), img)
    json_path = os.path.join(dest_dir, "scene.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(scene_dict, file, ensure_ascii=False, indent=4)


def add_background(img, background, foreground_mask):
    """Returns an image where one background is replaced with another

    Args:
        img (array): Image to replace background
        background (array): New background
        foreground_mask (array): The part of image that shall remain the same
    """
    if img.shape != background.shape:
        resize = A.Resize(*img.shape[:2])
        background = resize(image=background)["image"]
    img[foreground_mask == 0] = 0
    background[foreground_mask != 0] = 0
    return img + background


def merge_masks(masks):
    """Creates a single mask from a list of masks

    Args:
        masks (list): Masks to merge
    """
    return np.amax(masks, axis=0)


# def get_new_mask(orig_mask, foreground_mask):
# new_mask = orig_mask.copy()
## TODO hide parts that are hidden by foreground detalis
# new_mask[foreground_mask != 0] = 0
# return new_mask


def augment(
    scene_json_path,
    background_path,
    foreground_scene,
    dest,
    transform_background=None,
    transform_mask=None,
    p_drop=0.0,
):
    """Augment scene with background and foreground details in a correct way

    Args:
        scene_json_path (str): Path base scene
        background_path (str): Path to image with new background
        foreground_scene (str): Path to scene with details to be added in foreground
        dest (str): Path save new scene
        transform_background: Albumentations for background image
        transform_mask: Albumentations for visible objects mask. Is save for geometric transforms.
        p_drop (float): Probability to drop each detail in foreground
    """
    scene_dict, img, path_to_mask = load_scene(scene_json_path, add_mask_prefix="orig_")
    if foreground_scene:
        f_scene_dict, f_img, f_mask_to_path = load_scene(
            foreground_scene, add_mask_prefix="foreground_"
        )

        filtered_objects = []
        # TODO If a detali A was partially hidden by a detalil B on foreground
        # and the detail B is dropped then A shall be also dropped
        for obj in f_scene_dict["objects"]:
            if uniform(0, 1) > p_drop:  # FIXME???
                filtered_objects.append(obj)
        f_scene_dict["objects"] = filtered_objects
        f_visible_masks = [
            f_mask_to_path[obj["visible_mask"]] for obj in f_scene_dict["objects"]
        ]
        foreground_mask = merge_masks(f_visible_masks)
        assert img.size == f_img.size, "Scenes have different sizes"
        img = add_background(f_img, img, foreground_mask)

        # обновим оригинальные маски
        for obj in scene_dict["objects"]:
            visible_mask = path_to_mask[obj["visible_mask"]]
            visible_mask[foreground_mask != 0] = 0

        path_to_mask = {**path_to_mask, **f_mask_to_path}
        scene_dict["objects"] += f_scene_dict["objects"]

    if background_path:
        masks = [path_to_mask[obj["visible_mask"]] for obj in scene_dict["objects"]]
        mask = merge_masks(masks)

        background = cv2.imread(background_path)

        if transform_background:
            background = transform_background(image=background)["image"]

        if transform_mask:
            extra_mask = np.zeros_like(masks[0])
            extra_mask = transform_mask(image=extra_mask)["image"]
            apply_mask = merge_masks([mask, extra_mask])
        else:
            apply_mask = mask
        img = add_background(img, background, apply_mask)

    correct_and_save_scene(
        dest, scene_dict, img, path_to_mask, foreground_scene is not None
    )


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    transform_background = A.from_dict(cfg.BACKGROUND_TRANSFORM)
    transform_mask = A.from_dict(cfg.BACKGROUND_MASK_TRANSFORM)

    orig_scenes = sorted(Path(cfg.ORIG_SET).glob("scenes/**/*.json"))
    backgrounds = []
    if cfg.BACKGROUNDS:
        for ext in _IMG_EXT:
            backgrounds += list(Path(cfg.BACKGROUNDS).glob(f"**/*{ext}"))
        # if len(backgrounds) == 0:
        # raise ValueError(f"No backgrounds in {cfg.BACKGROUNDS} are found!")

    foregrounds = []
    foregrounds_swap_p = []
    foregrounds_drop_p = []
    for foreground in cfg.FOREGROUND_SETS:
        scenes = list(Path(foreground.DIR).glob("scenes/**/*.json"))
        foregrounds += scenes
        foregrounds_swap_p += [foreground.P_SWAP] * len(scenes)
        foregrounds_drop_p += [foreground.P_ITEM_DROP] * len(scenes)
    foregrounds_with_p = list(
        zip(foregrounds, foregrounds_swap_p, foregrounds_drop_p)
    )  # TODO rename??

    if len(backgrounds) == 0 and len(foregrounds) == 0:
        raise "There are no sources that can be used for augmentation!"

    if cfg.VAL_PART > 0:
        # FIXME Backgrounds and foregrounds must be splitted too!
        # But in the case of foregrounds there will still exists some leak of train data
        # to val, because they will be also splitted on train/val for their set separately
        train, val = train_test_split(
            orig_scenes, test_size=cfg.VAL_PART, random_state=cfg.SPLIT_SEED
        )
        it = zip(["train", "val"], [train, val])
    else:
        it = [("", orig_scenes)]

    for p, files_part in it:
        if p:
            print(f"Createing {p} part")
        dest = os.path.join(cfg.NEW_SET, p, "scenes")

        scenes_l, dest_l = [], []
        backgrounds_l, foregrounds_l = [], []
        swap_p_l, drop_p_l = [], []
        for i, scene in enumerate(files_part):
            scenes_l += [scene] * cfg.AUGS_PER_SCENE
            dest_l += [
                os.path.join(dest, f"{i}_{j}") for j in range(cfg.AUGS_PER_SCENE)
            ]
            backgrounds_l += (
                sample(backgrounds, cfg.AUGS_PER_SCENE)
                if backgrounds
                else [None] * cfg.AUGS_PER_SCENE
            )
            s = (
                sample(foregrounds_with_p, cfg.AUGS_PER_SCENE)
                if foregrounds_with_p
                else [(None, None, None)] * cfg.AUGS_PER_SCENE
            )
            foreground_s, swap_p, drop_p = zip(*s)
            foregrounds_l += foreground_s
            swap_p_l += swap_p
            drop_p_l += drop_p

        for i in range(len(scenes_l)):
            # Different sets have different relative distance. To minimize domain shift we swap prob should depending on distance
            if uniform(0, 1) < swap_p_l[i] if swap_p_l[i] else False:
                if foregrounds_l[i]:
                    scenes_l[i], foregrounds_l[i] = foregrounds_l[i], scenes_l[i]

        os.makedirs(dest)
        Parallel(n_jobs=cfg.JOBS)(
            delayed(augment)(
                scene_json_path=str(s),
                background_path=str(b) if b else None,
                foreground_scene=str(f) if f else None,
                dest=str(d),
                transform_background=transform_background,
                transform_mask=transform_mask,
                p_drop=p_drop,
            )
            for i, (s, b, f, d, p_drop) in enumerate(
                tqdm(
                    list(zip(scenes_l, backgrounds_l, foregrounds_l, dest_l, drop_p_l))
                )
            )
        )


if __name__ == "__main__":
    main()
