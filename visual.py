import csv
from email import header
from pathlib import Path

import matplotlib.pyplot as plt

from refer import REFER

DATA_DIR = Path("__file__").resolve().parent / "data"


def main(dataset: str, splitBy: str, ref_id: int, save_dir: str):
    assert dataset in ["refcoco", "refcoco+", "refcocog"]
    assert splitBy in ["unc", "google", "umd"]
    refer = REFER(str(DATA_DIR), dataset, splitBy)

    ref = refer.Refs[ref_id]
    plt.figure()
    refer.showRef(ref, seg_box="seg")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # save the figure
    plt.savefig(
        str(save_dir / f"{dataset}_{splitBy}_ref{ref_id}.png"),
        bbox_inches="tight",
        transparent=True,
    )


def find_ref_id(dataset="refcoco+", splitBy="unc", setName="train", idx: int = None):
    custom_dir = Path("/home/markhh/CODE/DEEP_LEARNING/novel_composition/data/custom")
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


def get(save_dir="nv10", dataset="refcoco+", setName="train", idx: int = None):
    return main(
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


if __name__ == "__main__":
    # main("refcoco+", "unc", ref_id=861)
    # main("refcoco+", "unc", ref_id=19794)
    # main("refcoco+", "unc", ref_id=48870)

    get("nv-tmp", "refcoco+", "val", 251944)
    get("nv-tmp", "refcoco+", "val", 255415)
    get("nv-tmp", "refcoco+", "val", 256032)
    get("nv-tmp", "refcoco+", "val", 258570)
    get("nv-tmp", "refcoco+", "val", 258726)
    get("nv-tmp", "refcoco+", "val", 258828)
    get("nv-tmp", "refcoco+", "val", 261452)
    get("nv-tmp", "refcoco+", "val", 252014)
    get("nv-tmp", "refcoco+", "val", 252129)
    get("nv-tmp", "refcoco+", "val", 253890)
    get("nv-tmp", "refcoco+", "val", 253988)
    get("nv-tmp", "refcoco+", "val", 256978)
