import matplotlib
from pathlib import Path
import json
from typing import *
import matplotlib.pyplot as plt
import numpy as np


def main():
    fp = Path("/home/markhh/Documents/fig_examples/refcoco_testB_iou.json")
    with fp.open("r") as f:
        data: List[float] = json.load(f)

    print(len(data))

    # plot histogram

    plt.hist(data, bins=10)
    # save figure
    plt.savefig("tmp.png")


def check():
    npz = "/home/markhh/Documents/fig_examples/refcoco_testA_mask.npz"
    data = np.load(npz, allow_pickle=True)
    print(data["arr_0"].shape)
    matrix = []
    for i in data:
        mask = data[i]
        mask = np.expand_dims(mask, axis=0)
        matrix.append(mask)

    matrix = np.concatenate(matrix, axis=0)
    print(matrix.shape)


if __name__ == "__main__":
    check()
