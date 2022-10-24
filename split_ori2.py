"""
Author: Mark H H 

This script is used to split the original refcoco/refcoco+/refcocog dataset into new splits based on uids file *_train_split_uids.json
"""

import copy
import json
import pickle
from pathlib import Path

from puts import print_cyan, print_green, print_red, print_yellow

from refer import REFER

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def revert_uid_map(uids_fp: str):
    uids_fp = Path(uids_fp)
    assert uids_fp.exists()

    with open(uids_fp, "r") as f:
        uids = json.load(f)

    reverse_map = {}

    for key in ["meta_support", "meta_query_ww", "meta_query_wp", "meta_query_pp"]:
        set_uids = uids[key]
        for uid in set_uids:
            reverse_map[uid] = key

    return reverse_map


def split_original(
    dataset: str, split_by: str, uids_fp: str, new_split_by: str, overwrite: bool
):
    assert dataset in ["refcoco", "refcoco+", "refcocog"]
    assert split_by in ["unc", "google", "umd"]
    ref_fp = DATA_DIR / dataset / f"refs({split_by}).p"
    assert ref_fp.exists()

    export_fp = DATA_DIR / dataset / f"refs({new_split_by}).p"
    if export_fp.exists():
        if overwrite:
            print_yellow(f"Overwriting '{export_fp}'...")
        else:
            print_yellow(f"'{export_fp}' already exists. Skip...")
            return

    uids_fp = Path(uids_fp)
    assert uids_fp.exists()

    reverse_map = revert_uid_map(uids_fp)

    with open(ref_fp, "rb") as f:
        refs: list = pickle.load(f)

    print(f"Loaded {len(refs)} refs from {ref_fp}")

    not_used_sent_counter = 0

    new_refs = []
    for ref in refs:
        ref_id = ref.get("ref_id")
        num_sent = len(ref.get("sent_ids"))
        if num_sent == 0:
            print(f"Ref {ref_id} has no sentences")
            continue

        for i in range(num_sent):
            uid = f"{ref_id}_{i}"
            new_split = reverse_map.get(uid)
            if not new_split:
                not_used_sent_counter += 1
                continue

            new_ref = copy.deepcopy(ref)
            new_ref["split"] = new_split
            new_ref["sent_ids"] = [ref["sent_ids"][i]]
            new_ref["sentences"] = [ref["sentences"][i]]
            new_refs.append(new_ref)

    print(f"Total {not_used_sent_counter} sentences are not used")
    print(f"Total {len(new_refs)} new refs are generated (with each contains one sent)")

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
        "meta_support",
        "meta_query_ww",
        "meta_query_wp",
        "meta_query_pp",
    ]:
        ref_ids = refer.getRefIds(split=split_name)
        print_cyan(f"{len(ref_ids)} refs are in split [{split_name}].")
        if len(ref_ids) == 0:
            print_red(f"No such split available: {split_name}")
            continue


def main(split_num: int):
    assert split_num in [1, 2, 3]

    split_dir = (
        ROOT.parent
        / "DEEP_LEARNING"
        / "novel_composition"
        / "data"
        / f"meta_split{split_num}"
    )

    split_original(
        dataset="refcoco",
        split_by="unc",
        uids_fp=split_dir / "refcoco_unc_train_split_uids.json",
        new_split_by=f"meta_split{split_num}",
        overwrite=False,
    )
    split_original(
        dataset="refcoco+",
        split_by="unc",
        uids_fp=split_dir / "refcoco+_unc_train_split_uids.json",
        new_split_by=f"meta_split{split_num}",
        overwrite=False,
    )
    split_original(
        dataset="refcocog",
        split_by="umd",
        uids_fp=split_dir / "refcocog_umd_train_split_uids.json",
        new_split_by=f"meta_split{split_num}",
        overwrite=False,
    )

    check(dataset="refcoco", split_by=f"meta_split{split_num}")
    check(dataset="refcoco+", split_by=f"meta_split{split_num}")
    check(dataset="refcocog", split_by=f"meta_split{split_num}")


if __name__ == "__main__":
    main(1)
    main(2)
    main(3)
