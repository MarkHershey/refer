from pathlib import Path
from pprint import pprint

from puts import print_green

from refer import REFER

DATA_DIR = Path("__file__").resolve().parent / "data"


def main(dataset: str, splitBy: str):
    assert dataset in ["refcoco", "refcoco+", "refcocog"]
    assert splitBy in ["unc", "google", "umd"]
    refer = REFER(str(DATA_DIR), dataset, splitBy)

    base_name = f"{dataset}_{splitBy}"

    # print stats about the given dataset
    print(f"\nDataset {dataset}_{splitBy} contains: ")
    ref_ids = refer.getRefIds()
    image_ids = refer.getImgIds()
    print(
        f"{len(refer.Sents)} expressions for {len(ref_ids)} refs in {len(image_ids)} images."
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
        split_name = f"{base_name}_{split}"
        csv_filepath = DATA_DIR / f"{split_name}.csv"
        csv_content_lines = ["ref_id,img_id, ref_exp_id, exp_id"]

        ref_ids = refer.getRefIds(split=split)
        print(f"{len(ref_ids)} refs are in split [{split_name}].")
        for ref_id in ref_ids[123:130]:
            print_green(f"ref_id: {ref_id}")
            ref = refer.Refs[ref_id]
            image_id = ref.get("image_id")
            print_green(f"img_id: {image_id}")
            for idx, sent in enumerate(ref["sentences"]):
                ref_exp_id = f"{ref_id}_{idx}"
                print_green(f"ref_exp_id: {ref_exp_id}")
                print(f"expression[{sent['sent_id']}]: {sent['sent']}")
            print()
        break


def peak():
    import json

    tmp = "data/refcoco/instances.json"
    with open(tmp, "r") as f:
        data = json.load(f)

    print(data.keys())
    images = data["images"]
    annotations = data["annotations"]
    print(len(images))
    print(len(annotations))
    print(images[0])
    print(annotations[0])


if __name__ == "__main__":
    # main("refcoco", "unc")
    # main("refcoco", "google")
    # main("refcoco+", "unc")
    # main("refcocog", "google")
    # main("refcocog", "umd")
    ...
