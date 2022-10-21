import csv
import json
import os
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import binary_dilation

from refer import REFER

# from matplotlib import lines, patches
# from matplotlib.patches import Polygon
# from skimage.measure import find_contours

result_dir = Path("/home/markhh/Documents/qualitative")
baseline_fp = result_dir / "refcoco_val_baseline_qual.json"
ours_fp = result_dir / "refcoco_val_meta_qual.json"
DATA_DIR = Path("__file__").resolve().parent / "data"
IMG_DIR = DATA_DIR / "images" / "mscoco" / "train2014"


def find_ref_id(
    dataset: str = "refcoco",
    splitBy: str = "unc",
    setName: str = "val",
    idx: int = None,
):
    """
    note that this idx starts from zero (with no offsets)
    """
    custom_dir = Path("/home/markhh/CODE/DEEP_LEARNING/novel_composition/data/custom")
    assert custom_dir.exists()
    custom_file = custom_dir / f"{dataset}_{splitBy}_{setName}.csv"
    assert custom_file.exists(), f"{custom_file}"

    with open(custom_file, "r") as f:
        csv_reader = list(csv.reader(f))
        header = csv_reader.pop(0)
        assert 0 <= idx < len(csv_reader)
        row = list(csv_reader)[idx]
        return int(row[2])


def get_image_dict(
    dataset: str = "refcoco",
    splitBy: str = "unc",
    ref_id: int = 0,
):
    assert dataset in ["refcoco", "refcoco+", "refcocog"]
    assert splitBy in ["unc", "google", "umd"]
    refer = REFER(str(DATA_DIR), dataset, splitBy)

    # ref = refer.Refs[ref_id]
    # img_id = ref.get("image_id")
    img_id = refer.getImgIds(ref_ids=[ref_id])[0]
    print("Image ID: ", img_id)
    img_dict = refer.Imgs[img_id]
    file_name = img_dict.get("file_name")
    height = img_dict.get("height")
    width = img_dict.get("width")
    print(f"Image file name: {file_name} ({width}x{height})")
    return img_dict


def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image * alpha + np.ones(image.shape) * (1 - alpha) * np.array(
            colors[object_id]
        )
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


def visual(
    dataset: str = "refcoco",
    splitBy: str = "unc",
    setName: str = "val",
    idx: int = None,
    pred: torch.Tensor = None,
    save_dir: str = "qualitative",
    gt: bool = False,
):
    ref_id = find_ref_id(dataset, splitBy, setName, idx)
    img_dict = get_image_dict(dataset, splitBy, ref_id)
    img_file_name = img_dict.get("file_name")

    image_path = IMG_DIR / img_file_name

    img = Image.open(image_path).convert("RGB")
    img_ndarray = np.array(img)  # (orig_h, orig_w, 3); for visualization
    original_w, original_h = img.size  # PIL .size returns width first and height second

    output = pred  # (1, 1, 480, 480)
    output = F.interpolate(
        output.float(), (original_h, original_w)
    )  # 'nearest'; resize to the original image size
    output = output.squeeze()  # (orig_h, orig_w)
    output = output.cpu().data.numpy()  # (orig_h, orig_w)

    output = output.astype(np.uint8)  # (orig_h, orig_w), np.uint8
    # Overlay the mask on the image
    visualization = overlay_davis(img_ndarray, output)  # red
    visualization = Image.fromarray(visualization)

    # show the visualization
    # visualization.show()

    # Save the visualization
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = "gt" if gt else "pred"
    save_path = str(save_dir / f"{dataset}_{splitBy}_ref{ref_id}_idx{idx}_{suffix}.png")
    visualization.save(save_path)
    print(f"Saved image to: {save_path}")


def main():
    with baseline_fp.open() as f:
        baseline = json.load(f)

    n_samples = len(baseline)
    assert len(baseline[0]) == 4

    for i in baseline:
        assert len(i) == 4
        idx = i[0]
        iou = i[1]
        pred = torch.tensor(i[2])
        gt = torch.tensor(i[3])
        pred.unsqueeze_(0).unsqueeze_(0)
        print(pred.shape, gt.shape)
        visual(idx=idx, pred=pred)
        return


if __name__ == "__main__":
    main()
