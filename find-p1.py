import csv
import json
import os
from pathlib import Path
from typing import *

import nltk
import matplotlib.pyplot as plt
from refer import REFER
from tqdm import tqdm
from puts import print_green, print_cyan, print_red, print_yellow

DATA_DIR = Path("__file__").resolve().parent / "data"
novel_composition_dir = Path("/home/markhh/CODE/DEEP_LEARNING/novel_composition")


coco_val_snc = novel_composition_dir / "coco-val-snc.json"
coco_val_sents = novel_composition_dir / "data" / "c_finetune_refcoco_val.json"
coco_train_sents = novel_composition_dir / "data" / "c_finetune_refcoco_train.json"

working_dir = Path("working_dir")
working_dir.mkdir(exist_ok=True)

refer = REFER(str(DATA_DIR), "refcoco", "unc")


def save_fig(dataset: str, splitBy: str, ref_id: int, save_dir: str = working_dir):
    out_file = save_dir / f"{dataset}_{splitBy}_ref{ref_id}.png"
    if out_file.exists():
        return str(out_file)

    ref = refer.Refs[ref_id]
    plt.figure()
    refer.showRef(ref, seg_box="seg")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # save the figure
    plt.savefig(
        str(out_file),
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()
    return str(out_file)


def find_ref_id(dataset="refcoco", splitBy="unc", setName="train", idx: int = None):
    """
    Note that this func assumes that idx supplied is after offset
    """
    custom_dir = novel_composition_dir / "data/custom"
    assert custom_dir.exists()
    custom_file = custom_dir / f"{dataset}_{splitBy}_{setName}.csv"
    assert custom_file.exists(), f"{custom_file}"

    with open(custom_file, "r") as f:
        csv_reader = list(csv.reader(f))
        header = csv_reader.pop(0)
        for row in csv_reader:
            if int(row[0]) == idx:
                return int(row[2])

    raise ValueError(f"Cannot find {idx} in {custom_file}")


def get(save_dir=working_dir, dataset="refcoco", setName="val", idx: int = None):
    return save_fig(
        dataset=dataset,
        splitBy="umd" if dataset == "refcocog" else "unc",
        save_dir=save_dir,
        ref_id=find_ref_id(
            dataset=dataset,
            splitBy="umd" if dataset == "refcocog" else "unc",
            setName=setName,
            idx=idx,
        ),
    )


def main():
    with open(coco_val_snc, "r") as f:
        ncs = json.load(f)

    with open(coco_val_sents, "r") as f:
        sents = json.load(f)

    md_data = []

    # for nc in ncs:
    for nc in tqdm(ncs):
        idx, i, j = nc
        sent = sents[str(idx)]
        if len(sent.split()) > 6:
            continue
        if " " in i or " " in j:
            continue
        tags = nltk.pos_tag([i, j], tagset="universal")
        if tags[0][1] != "ADJ" or tags[1][1] != "NOUN":
            continue

        new_nc = nc[:]
        new_nc.append(sent)
        new_nc.append(get(dataset="refcoco", setName="val", idx=idx))
        md_data.append(new_nc)

    md_lines = ["| idx | i | j | sent | img |", "| :-: | :-: | :-: | :-: | :-: |"]
    print(f"{len(md_data)} candidates")

    for nc in md_data:
        md_lines.append(f"| {nc[0]} | {nc[1]} | {nc[2]} | {nc[3]} | ![]({nc[4]}) |")

    with open("find-p1.md", "w") as f:
        f.write("\n".join(md_lines))


def find_train(word: str, length_limit: int = 6, max_count: int = 30):
    with coco_train_sents.open("r") as f:
        sents = json.load(f)

    md_lines = ["| idx | sent | img |", "| :-: | :-: | :-: |"]

    for idx, sent in sents.items():
        if len(sent.split()) > length_limit:
            print_yellow(f"skip long sent {idx}: {sent}")
            continue
        if word in sent.split():
            fig = get(dataset="refcoco", setName="train", idx=int(idx))
            item = f"| {idx} | {sent} | ![]({fig}) |"
            md_lines.append(item)

        if len(md_lines) > max_count + 2:
            print_yellow(f"Early stop: Found {len(md_lines) - 2} items")
            break

    with open(f"find-p1-train-{word}.md", "w") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    ...
    # main()
    find_train(word="yellow")
    find_train(word="laptop")
