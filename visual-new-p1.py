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
from tqdm import tqdm
from puts import print_green, print_cyan, print_red

# from matplotlib import lines, patches
# from matplotlib.patches import Polygon
# from skimage.measure import find_contours

result_dir = Path("/home/markhh/Downloads/fig_examples")
# baseline_fp = result_dir / "refcoco_val_baseline_qual.json"
baseline_bad_fp = result_dir / "refcoco_val_iou0.2.json"
baseline_gud_fp = result_dir / "refcoco_val_iou0.9.json"


# ours_fp = result_dir / "refcoco_val_meta_qual.json"
DATA_DIR = Path("__file__").resolve().parent / "data"
IMG_DIR = DATA_DIR / "images" / "mscoco" / "train2014"
custom_dir = Path("/home/markhh/CODE/DEEP_LEARNING/novel_composition/data/custom")
assert custom_dir.exists()


refer = REFER(str(DATA_DIR), "refcoco", "unc")


def find_ref_id(
    dataset: str = "refcoco",
    splitBy: str = "unc",
    setName: str = "val",
    idx: int = None,
):
    """
    note that this idx starts from zero (with no offsets)
    """
    custom_file = custom_dir / f"{dataset}_{splitBy}_{setName}.csv"
    assert custom_file.exists(), f"{custom_file}"

    with open(custom_file, "r") as f:
        csv_reader = list(csv.reader(f))
        header = csv_reader.pop(0)
        assert 0 <= idx < len(csv_reader)
        row = list(csv_reader)[idx]
        return int(row[2])


def find_custom_sample_line(
    dataset: str = "refcoco",
    splitBy: str = "unc",
    setName: str = "val",
    idx: int = None,
):
    """
    note that this idx starts from zero (with no offsets)
    """
    custom_file = custom_dir / f"{dataset}_{splitBy}_{setName}.csv"
    assert custom_file.exists(), f"{custom_file}"

    with open(custom_file, "r") as f:
        csv_reader = list(csv.reader(f))
        header = csv_reader.pop(0)
        assert 0 <= idx < len(csv_reader)
        row = list(csv_reader)[idx]
        return row


def get_image_dict(
    dataset: str = "refcoco",
    splitBy: str = "unc",
    ref_id: int = 0,
):
    assert dataset in ["refcoco", "refcoco+", "refcocog"]
    assert splitBy in ["unc", "google", "umd"]
    # refer = REFER(str(DATA_DIR), dataset, splitBy)

    # ref = refer.Refs[ref_id]
    # img_id = ref.get("image_id")
    img_id = refer.getImgIds(ref_ids=[ref_id])[0]
    # print("Image ID: ", img_id)
    img_dict = refer.Imgs[img_id]
    file_name = img_dict.get("file_name")
    height = img_dict.get("height")
    width = img_dict.get("width")
    # print(f"Image file name: {file_name} ({width}x{height})")
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
    save_dir: str = "visual-p1",
    gt: bool = False,
    name: str = "",
) -> Path:
    try:
        ref_id = find_ref_id(dataset, splitBy, setName, idx)
        img_dict = get_image_dict(dataset, splitBy, ref_id)
        img_file_name = img_dict.get("file_name")

        # Save the visualization
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = "gt" if gt else f"pred_{name}"
        save_path = str(
            save_dir / f"{dataset}_{splitBy}_ref{ref_id}_idx{idx}_{suffix}.png"
        )
        if Path(save_path).exists():
            # print(f"{save_path} already exists. Skipping.")
            return Path(save_path)

        image_path = IMG_DIR / img_file_name

        img = Image.open(image_path).convert("RGB")
        img_ndarray = np.array(img)  # (orig_h, orig_w, 3); for visualization
        (
            original_w,
            original_h,
        ) = img.size  # PIL .size returns width first and height second

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

        visualization.save(save_path)
        # print_green(f"Saved image to: {save_path}")
        # print()
        return Path(save_path)
    except Exception as e:
        print_red(f"Error: {e}")


def main():
    test_split_fp = custom_dir / "refcoco_unc_val_split_idxes.json"
    assert test_split_fp.exists()

    with open(test_split_fp, "r") as f:
        split_idxes = json.load(f)
        keys = list(split_idxes.keys())
        print(f"keys: {keys}")
        sww = set(split_idxes.get("sww"))
        swp = set(split_idxes.get("swp"))
        spp = set(split_idxes.get("spp"))
        snv = spp | swp | sww

    with baseline_bad_fp.open() as f:
        bad = json.load(f)

    with baseline_gud_fp.open() as f:
        gud = json.load(f)

    n_samples = len(bad)
    assert len(bad[0]) == 3

    ious = []
    sentences = []
    vi_paths = []

    ref_ids = []
    info_lines = []
    nc_lines_lst = []

    export_data = []

    nc_lookup_fp = Path(
        "/home/markhh/CODE/DEEP_LEARNING/novel_composition/coco-val-snc.json"
    )
    with nc_lookup_fp.open() as f:
        nc_lookup = json.load(f)

    for model in ["bad", "gud"]:

        for i in tqdm(eval(model)):
            assert len(i) == 3
            idx = i[0]
            iou = i[1]
            pred = i[2]
            pred = torch.tensor(pred).unsqueeze_(0)
            assert pred.shape == (1, 1, 480, 480)

            sample = find_custom_sample_line(idx=idx)
            idx_w_offset = int(sample[0])
            ref_id = int(sample[2])

            # if idx_w_offset not in snv:
            #     continue

            ious.append(iou)
            sent = str(sample[5])
            sentences.append(sent)

            # info_line = f'{idx} {idx_w_offset} {ref_id} {"sww" if idx_w_offset in sww else "---"} {"swp" if idx_w_offset in swp else "---"} {"spp" if idx_w_offset in spp else "---"}'
            info_line = f'{"sww" if idx_w_offset in sww else "---"} {"swp" if idx_w_offset in swp else "---"} {"spp" if idx_w_offset in spp else "---"}'
            # print_cyan(info_line)
            nc_lines = []

            ref_ids.append(ref_id)
            info_lines.append(info_line)
            ncs = [x for x in nc_lookup if x[0] == idx_w_offset]
            for j in ncs:
                nc_line = f"[{j[1]}] [{j[2]}]"
                nc_lines.append(nc_line)
            nc_lines_lst.append(nc_lines)

            vi_path = visual(idx=idx, pred=pred, gt=False, name=model)
            vi_paths.append(vi_path)

            data = [
                idx,
                idx_w_offset,
                ref_id,
                info_line,
                iou,
                sent,
                str(vi_path),
                nc_lines,
            ]
            export_data.append(data)

    # save the data
    save_dir = Path("visual-p1.json")
    with save_dir.open("w") as f:
        json.dump(export_data, f, indent=4)


def findX(lst, X: str):
    candidates = []
    for i in lst:
        ncs = i[7]
        for j in ncs:
            if X in j:
                candidates.append(i)
                break
    if len(candidates) == 0:
        return None
    else:
        return candidates


def find_p1():
    data_fp = Path("visual-p1.json")
    with data_fp.open() as f:
        data = json.load(f)

    candidates = []

    bad = [x for x in data if x[4] < 0.5 and "s" in x[3]]
    gud = [x for x in data if x[4] >= 0.5 and "s" not in x[3]]

    print("bad", len(bad))
    print("gud", len(gud))

    for i in tqdm(bad):
        ncs = i[7]
        for nc in ncs:
            idx = nc.find("] [")
            if idx == -1:
                print_red("Unexpected format")
                continue
            A = nc[: idx + 1]
            B = nc[idx + 2 :]

            M = findX(gud, A)
            if M is None:  # not found
                print(f"not found: {A}")
                continue
            N = findX(gud, B)
            if N is None:  # not found
                print(f"not found: {B}")
                continue

            _tmp = dict(H=i, M=M, N=N)
            candidates.append(_tmp)

    print("candidates", len(candidates))
    with open("candidates.json", "w") as f:
        json.dump(candidates, f, indent=4)
        print("Saved to candidates.json")


if __name__ == "__main__":
    ...
    # main()
    find_p1()
