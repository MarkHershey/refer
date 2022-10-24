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

    with open(uids_fp, "r") as f:
        uid_splits = json.load(f)

    with open(ref_fp, "rb") as f:
        refs: list = pickle.load(f)

    print(f"Loaded {len(refs)} refs from {ref_fp}")

    uid_ref_map = {}
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
            uid_ref_map[uid] = new_ref

    new_refs = []
    for new_split in [
        "meta_support",
        "meta_query_sww",
        "meta_query_swp",
        "meta_query_spp",
        "meta_support_sww",
        "meta_support_swp",
        "meta_support_spp",
    ]:
        set_uids = uid_splits.get(new_split, [])
        print(f" >>>>>>>>>>   Split [{new_split}] has {len(set_uids)} uids")
        for uid in set_uids:
            new_ref = copy.deepcopy(uid_ref_map.get(uid))
            if new_ref is None:
                print_red(f"UID {uid} not found in original refs")
                continue
            new_ref["split"] = new_split
            new_refs.append(new_ref)

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
        "meta_query_sww",
        "meta_query_swp",
        "meta_query_spp",
        "meta_support_sww",
        "meta_support_swp",
        "meta_support_spp",
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
        new_split_by=f"meta_split{split_num}s",
        overwrite=False,
    )
    split_original(
        dataset="refcoco+",
        split_by="unc",
        uids_fp=split_dir / "refcoco+_unc_train_split_uids.json",
        new_split_by=f"meta_split{split_num}s",
        overwrite=False,
    )
    split_original(
        dataset="refcocog",
        split_by="umd",
        uids_fp=split_dir / "refcocog_umd_train_split_uids.json",
        new_split_by=f"meta_split{split_num}s",
        overwrite=False,
    )

    check(dataset="refcoco", split_by=f"meta_split{split_num}s")
    check(dataset="refcoco+", split_by=f"meta_split{split_num}s")
    check(dataset="refcocog", split_by=f"meta_split{split_num}s")


if __name__ == "__main__":
    main(1)
    main(2)
    main(3)
