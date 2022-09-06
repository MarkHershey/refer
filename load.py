import matplotlib.pyplot as plt
import numpy as np

from refer import REFER

from skimage import io


data_root = "./data"  # contains refclef, refcoco, refcoco+, refcocog and images
dataset = "refcoco"
splitBy = "unc"
refer = REFER(data_root, dataset, splitBy)

# print stats about the given dataset
print("dataset [%s_%s] contains: " % (dataset, splitBy))
ref_ids = refer.getRefIds()
image_ids = refer.getImgIds()
print(
    "%s expressions for %s refs in %s images."
    % (len(refer.Sents), len(ref_ids), len(image_ids))
)

print("\nAmong them:")
if dataset == "refclef":
    if splitBy == "unc":
        splits = ["train", "val", "testA", "testB", "testC"]
    else:
        splits = ["train", "val", "test"]
elif dataset == "refcoco":
    splits = ["train", "val", "test"]
elif dataset == "refcoco+":
    splits = ["train", "val", "test"]
elif dataset == "refcocog":
    splits = ["train", "val"]  # we don't have test split for refcocog right now.

for split in splits:
    ref_ids = refer.getRefIds(split=split)
    print("%s refs are in split [%s]." % (len(ref_ids), split))
