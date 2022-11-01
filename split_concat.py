"""
Author: Mark H H 

This script is used to concat aligned support-query sets
"""

import copy
import pickle
from pathlib import Path
from puts import print_cyan, print_green, print_red, print_yellow

from refer import REFER

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def concat(dataset: str, split_by: str, new_split_by: str, overwrite: bool):
    assert dataset in ["refcoco", "refcoco+", "refcocog"]
    # assert split_by in ["unc", "google", "umd"]
    ref_fp = DATA_DIR / dataset / f"refs({split_by}).p"
    assert ref_fp.exists()
    print_cyan(f"Concatenating {ref_fp} ...")

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

    refs_dict = {
        "meta_support": [],
        "meta_query_sww": [],
        "meta_query_swp": [],
        "meta_query_spp": [],
        "meta_support_sww": [],
        "meta_support_swp": [],
        "meta_support_spp": [],
    }
    for ref in refs:
        ref_id = ref.get("ref_id")
        num_sent = len(ref.get("sent_ids"))
        assert num_sent == 1

        for i in range(num_sent):
            uid = f"{ref_id}_{i}"
            _split = ref["split"]
            assert _split in refs_dict
            refs_dict[_split].append(ref)

    support_refs = [
        *refs_dict["meta_support_sww"],
        *refs_dict["meta_support_swp"],
        *refs_dict["meta_support_spp"],
    ]
    query_refs = [
        *refs_dict["meta_query_sww"],
        *refs_dict["meta_query_swp"],
        *refs_dict["meta_query_spp"],
    ]

    new_refs = []

    for ref in support_refs:
        new_ref = copy.deepcopy(ref)
        new_ref["split"] = "meta_support"
        new_refs.append(new_ref)
    for ref in query_refs:
        new_ref = copy.deepcopy(ref)
        new_ref["split"] = "meta_query"
        new_refs.append(new_ref)

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
        "meta_query",
    ]:
        ref_ids = refer.getRefIds(split=split_name)
        print_cyan(f"{len(ref_ids)} refs are in split [{split_name}].")
        if len(ref_ids) == 0:
            print_red(f"No such split available: {split_name}")
            continue


def main(split_num: int):

    concat(
        dataset="refcoco",
        split_by=f"meta_split{split_num}s",
        new_split_by=f"meta_split{split_num}s_concat",
        overwrite=False,
    )
    concat(
        dataset="refcoco+",
        split_by=f"meta_split{split_num}s",
        new_split_by=f"meta_split{split_num}s_concat",
        overwrite=False,
    )
    concat(
        dataset="refcocog",
        split_by=f"meta_split{split_num}s",
        new_split_by=f"meta_split{split_num}s_concat",
        overwrite=False,
    )

    check(dataset="refcoco", split_by=f"meta_split{split_num}s_concat")
    check(dataset="refcoco+", split_by=f"meta_split{split_num}s_concat")
    check(dataset="refcocog", split_by=f"meta_split{split_num}s_concat")


if __name__ == "__main__":
    main(1)
