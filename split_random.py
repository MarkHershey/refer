"""
Author: Mark H H 

This script is used to split the original refcoco/refcoco+/refcocog dataset into new splits randomly
"""

import copy
import json
import pickle
from pathlib import Path
import random
from puts import print_cyan, print_green, print_red, print_yellow

from refer import REFER

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def split_random(
    dataset: str, split_by: str, new_split_by: str, overwrite: bool, seed: int = 0
):
    assert dataset in ["refcoco", "refcoco+", "refcocog"]
    assert split_by in ["unc", "google", "umd"]
    ref_fp = DATA_DIR / dataset / f"refs({split_by}).p"
    assert ref_fp.exists()

    random.seed(seed)
    print(f"Random seed: {seed}")

    export_fp = DATA_DIR / dataset / f"refs({new_split_by}).p"
    if export_fp.exists():
        if overwrite:
            print_yellow(f"Overwriting '{export_fp}'...")
        else:
            print_yellow(f"'{export_fp}' already exists. Skip...")
            return

    with open(ref_fp, "rb") as f:
        refs: list = pickle.load(f)

    print(f"Loaded {len(refs)} refs from {ref_fp}")

    refs_flatten = []
    for ref in refs:
        ref_id = ref.get("ref_id")
        num_sent = len(ref.get("sent_ids"))
        if num_sent == 0:
            print(f"Ref {ref_id} has no sentences")
            continue

        for i in range(num_sent):
            uid = f"{ref_id}_{i}"

            new_ref = copy.deepcopy(ref)
            # new_ref["split"] = new_split
            new_ref["sent_ids"] = [ref["sent_ids"][i]]
            new_ref["sentences"] = [ref["sentences"][i]]
            refs_flatten.append(new_ref)

    random.shuffle(refs_flatten)
    split_idx = int(len(refs_flatten) * 0.6)
    support_random = refs_flatten[:split_idx]
    query_random = refs_flatten[split_idx:]

    new_refs = []

    for ref in support_random:
        ref["split"] = "meta_support_random"
        new_refs.append(ref)
    for ref in query_random:
        ref["split"] = "meta_query_random"
        new_refs.append(ref)

    with export_fp.open("wb") as f:
        pickle.dump(new_refs, f)
        print_green(f"Saved to '{export_fp}'")

    return None


def check(dataset: str, split_by: str):
    print(f"\nRunning check for {dataset} ({split_by}) ...")
    print(f"Dataset {dataset}_{split_by} contains: ")
    refer = REFER(str(DATA_DIR), dataset, split_by)
    ref_ids = refer.getRefIds()
    image_ids = refer.getImgIds()
    print_cyan(
        f"{len(refer.Sents)} expressions for {len(ref_ids)} refs in {len(image_ids)} images."
    )
    print("Among them:")
    for split_name in [
        "meta_support_random",
        "meta_query_random",
    ]:
        ref_ids = refer.getRefIds(split=split_name)
        print_cyan(f"{len(ref_ids)} refs are in split [{split_name}].")
        if len(ref_ids) == 0:
            print_red(f"No such split available: {split_name}")
            continue


def main(seed: int):

    split_random(
        dataset="refcoco",
        split_by="unc",
        new_split_by=f"meta_split_rand{seed}",
        overwrite=False,
        seed=seed,
    )
    split_random(
        dataset="refcoco+",
        split_by="unc",
        new_split_by=f"meta_split_rand{seed}",
        overwrite=False,
        seed=seed,
    )
    split_random(
        dataset="refcocog",
        split_by="umd",
        new_split_by=f"meta_split_rand{seed}",
        overwrite=False,
        seed=seed,
    )

    check(dataset="refcoco", split_by=f"meta_split_rand{seed}")
    check(dataset="refcoco+", split_by=f"meta_split_rand{seed}")
    check(dataset="refcocog", split_by=f"meta_split_rand{seed}")


if __name__ == "__main__":
    main(seed=0)
