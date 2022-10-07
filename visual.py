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
    custom_dir = Path("")
    assert custom_dir.exists()
    custom_file = custom_dir / f"{dataset}_{splitBy}_{setName}.csv"
    assert custom_file.exists()

    with open(custom_file, "r") as f:
        csv_reader = list(csv.reader(f))
        header = csv_reader.pop(0)
        for row in csv_reader:
            if int(row[0]) == idx:
                return int(row[2])

    raise ValueError(f"Cannot find {idx} in {custom_file}")


if __name__ == "__main__":
    # main("refcoco+", "unc", ref_id=861)
    # main("refcoco+", "unc", ref_id=19794)
    # main("refcoco+", "unc", ref_id=48870)

    main(
        "refcoco+",
        "unc",
        save_dir="nv1",
        ref_id=find_ref_id(setName="train", idx=135078),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv1",
        ref_id=find_ref_id(setName="train", idx=133567),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv1",
        ref_id=find_ref_id(setName="val", idx=252836),
    )

    main(
        "refcoco+",
        "unc",
        save_dir="nv2",
        ref_id=find_ref_id(setName="train", idx=137456),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv2",
        ref_id=find_ref_id(setName="train", idx=138073),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv2",
        ref_id=find_ref_id(setName="val", idx=256936),
    )

    main(
        "refcoco+",
        "unc",
        save_dir="nv3",
        ref_id=find_ref_id(setName="train", idx=194914),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv3",
        ref_id=find_ref_id(setName="train", idx=166748),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv3",
        ref_id=find_ref_id(setName="val", idx=253252),
    )

    main(
        "refcoco+",
        "unc",
        save_dir="nv4",
        ref_id=find_ref_id(setName="train", idx=173636),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv4",
        ref_id=find_ref_id(setName="train", idx=172401),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv4",
        ref_id=find_ref_id(setName="val", idx=260526),
    )

    main(
        "refcoco+",
        "unc",
        save_dir="nv5",
        ref_id=find_ref_id(setName="train", idx=162776),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv5",
        ref_id=find_ref_id(setName="train", idx=167181),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv5",
        ref_id=find_ref_id(setName="train", idx=184575),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv5",
        ref_id=find_ref_id(setName="val", idx=258718),
    )

    main(
        "refcoco+",
        "unc",
        save_dir="nv6",
        ref_id=find_ref_id(setName="train", idx=205395),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv6",
        ref_id=find_ref_id(setName="train", idx=235836),
    )
    main(
        "refcoco+",
        "unc",
        save_dir="nv6",
        ref_id=find_ref_id(setName="val", idx=254213),
    )
