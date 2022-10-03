from refer import REFER
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

DATA_DIR = Path("__file__").resolve().parent / "data"


def main(dataset: str, splitBy: str, ref_id: int):
    assert dataset in ["refcoco", "refcoco+", "refcocog"]
    assert splitBy in ["unc", "google", "umd"]
    refer = REFER(str(DATA_DIR), dataset, splitBy)

    ref = refer.Refs[ref_id]
    plt.figure()
    refer.showRef(ref, seg_box="seg")
    # save the figure
    plt.savefig(f"{dataset}_{splitBy}_ref{ref_id}.png")


if __name__ == "__main__":
    main("refcoco+", "unc", ref_id=861)
    main("refcoco+", "unc", ref_id=19794)
    main("refcoco+", "unc", ref_id=48870)
