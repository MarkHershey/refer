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
from puts import print_green, print_cyan, print_red, print_yellow

# from matplotlib import lines, patches
# from matplotlib.patches import Polygon
# from skimage.measure import find_contours


def init(
    setName: str = "val",
):
    global result_dir, baseline_fp, ours_fp, DATA_DIR, IMG_DIR, custom_dir, refer
    result_dir = Path("/home/markhh/Documents/qualitative_results")
    baseline_fp = result_dir / f"refcoco_{setName}_baseline_qual.json"
    ours_fp = result_dir / f"refcoco_{setName}_meta_qual.json"

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
    name: str = "",
) -> Path:
    try:
        ref_id = find_ref_id(dataset, splitBy, setName, idx)
        img_dict = get_image_dict(dataset, splitBy, ref_id)
        img_file_name = img_dict.get("file_name")

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

        # Save the visualization
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = "gt" if gt else f"pred_{name}"
        save_path = str(
            save_dir / f"{dataset}_{splitBy}_ref{ref_id}_idx{idx}_{suffix}.png"
        )
        visualization.save(save_path)
        print_green(f"Saved image to: {save_path}")
        print()
        return Path(save_path)
    except Exception as e:
        print_red(f"Error: {e}")


def main(setName="val"):
    assert setName in ["val", "testA", "testB"]
    test_split_fp = custom_dir / f"refcoco_unc_{setName}_split_idxes.json"
    assert test_split_fp.exists()

    with open(test_split_fp, "r") as f:
        split_idxes = json.load(f)
        keys = list(split_idxes.keys())
        print(f"keys: {keys}")
        sww = set(split_idxes.get("sww"))
        swp = set(split_idxes.get("swp"))
        spp = set(split_idxes.get("spp"))
        snv = spp | swp | sww

    with baseline_fp.open() as f:
        baseline = json.load(f)

    with ours_fp.open() as f:
        ours = json.load(f)

    n_samples = len(baseline)
    print(f"n_samples: {n_samples}")
    assert len(baseline[0]) == 4

    ours_ious = []
    base_ious = []
    captions = []
    gt_paths = []
    ours_paths = []
    base_paths = []
    ref_ids = []
    info_lines = []
    nc_lines_lst = []

    nc_lookup_fp = Path(
        f"/home/markhh/CODE/DEEP_LEARNING/novel_composition/coco-{setName}-snc.json"
    )
    with nc_lookup_fp.open() as f:
        nc_lookup = json.load(f)

    for model in ["baseline", "ours"]:
        isOurs = model == "ours"

        for i in eval(model):
            assert len(i) == 4
            idx = i[0]
            iou = i[1]
            pred = i[2]
            gt = i[3]

            sample = find_custom_sample_line(setName=setName, idx=idx)
            idx_w_offset = int(sample[0])
            ref_id = int(sample[2])
            if idx_w_offset not in snv:
                print_yellow(f"idx_w_offset {idx_w_offset} not in snv")
                continue

            if isOurs:
                ours_ious.append(iou)
                sent = str(sample[5])
                captions.append(sent)
            else:
                base_ious.append(iou)

            info_line = f'{idx} {idx_w_offset} {ref_id} {"sww" if idx_w_offset in sww else "---"} {"swp" if idx_w_offset in swp else "---"} {"spp" if idx_w_offset in spp else "---"}'
            print_cyan(info_line)
            nc_lines = []

            ref_ids.append(ref_id)
            if isOurs:
                info_lines.append(info_line)
                ncs = [x for x in nc_lookup if x[0] == idx_w_offset]
                for j in ncs:
                    nc_line = f"[{j[1]}] [{j[2]}]"
                    nc_lines.append(nc_line)
                nc_lines_lst.append(nc_lines)

            pred = torch.tensor(pred)
            gt = torch.tensor(gt)
            pred.unsqueeze_(0).unsqueeze_(0)
            gt.unsqueeze_(0).unsqueeze_(0)
            # print(pred.shape, gt.shape)
            if isOurs:
                gt_p = visual(setName=setName, idx=idx, pred=gt, gt=True)
                ours_p = visual(
                    setName=setName, idx=idx, pred=pred, gt=False, name="ours"
                )
                gt_paths.append(gt_p)
                ours_paths.append(ours_p)
            else:
                base_p = visual(
                    setName=setName, idx=idx, pred=pred, gt=False, name="base"
                )
                base_paths.append(base_p)

    markdown_lines = [
        "| Sentence | nc | GT | Ours | Base | Base-IOU | Ours-IOU | delta |"
    ]
    markdown_lines.append(
        "| :--- | :-- | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for i in range(len(gt_paths)):
        sent = captions[i]
        gt_p = gt_paths[i].name
        ours_p = ours_paths[i].name
        base_p = base_paths[i].name
        ours_iou = ours_ious[i]
        base_iou = base_ious[i]
        delta = "+" + str(round(ours_iou - base_iou, 2))
        ours_iou = str(round(ours_iou, 2))
        base_iou = str(round(base_iou, 2))
        nc_lines = nc_lines_lst[i]

        markdown_lines.append(
            f"| {ref_ids[i]} {sent} | {nc_lines} | ![{i}]({gt_p}) | ![{i}]({ours_p}) |  ![{i}]({base_p}) | {base_iou} | {ours_iou} | {delta} |"
        )

    # markdown_lines.append(" ")
    # markdown_lines.append(" ")
    # for i in range(len(info_lines)):
    #     info_line = info_lines[i]
    #     nc_lines = nc_lines_lst[i]
    #     markdown_lines.append(f"-   {info_line}")
    #     for j in nc_lines:
    #         markdown_lines.append(f"    -   {j}")

    with open(f"qualitative/README-{setName}.md", "w") as f:
        f.write("\n".join(markdown_lines))


if __name__ == "__main__":
    # init(setName="val")
    # main(setName="val")
    init()
    print(IMG_DIR / get_image_dict(ref_id=7249)["file_name"])
    print(IMG_DIR / get_image_dict(ref_id=37756)["file_name"])
    print(IMG_DIR / get_image_dict(ref_id=18295)["file_name"])
    print(IMG_DIR / get_image_dict(ref_id=33292)["file_name"])
    print(IMG_DIR / get_image_dict(ref_id=38781)["file_name"])
