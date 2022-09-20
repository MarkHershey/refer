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
        splits = ["train", "val", "testA", "testB"]
    elif dataset == "refcoco+":
        splits = ["train", "val", "testA", "testB"]
    elif dataset == "refcocog":
        splits = [
            "train",
            "val",
            "test",
        ]  # we don't have test split for refcocog right now.

    for split in splits:
        split_name = f"{base_name}_{split}"
        csv_filepath = DATA_DIR / f"{split_name}.csv"
        csv_content_lines = ["uid,ref_id,img_id,sent_id,sent,bbox"]

        ref_ids = refer.getRefIds(split=split)
        print(f"{len(ref_ids)} refs are in split [{split_name}].")
        if len(ref_ids) == 0:
            print(f"No such split available: {split_name}")
            continue

        for ref_id in ref_ids:
            print_green(f"ref_id: {ref_id}")
            bbox = refer.getRefBox(ref_id)
            bbox_str = str(bbox)

            ref = refer.Refs[ref_id]
            img_id = ref.get("image_id")
            print_green(f"img_id: {img_id}")
            print_green(f"ref: {ref}")

            mask = refer.getMask(refer.loadRefs(ref_id)[0])["mask"]
            print_green(f"mask: {mask}")

            for idx, sent in enumerate(ref["sentences"]):
                uid = f"{ref_id}_{idx}"
                sent_id = sent["sent_id"]
                sent = sent["sent"]
                new_line = f'{uid},{ref_id},{img_id},{sent_id},"{sent}","{bbox_str}"'
                csv_content_lines.append(new_line)
            break

        # with open(csv_filepath, "w") as f:
        #     f.write("\n".join(csv_content_lines))
        #     f.close()
        # print_green(f"Saved to {csv_filepath}")

    print("\n==================================================================\n")


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


def peak2():
    import pickle

    tmp = "data/refcoco/refs(unc).p"
    with open(tmp, "rb") as f:
        data: list = pickle.load(f)

    print_green(f"Length: {len(data)}")
    ref = data[0]
    print(ref)
    print_green(list(ref.keys()))


if __name__ == "__main__":
    # main("refcoco", "unc")
    # main("refcoco+", "unc")
    # main("refcocog", "google")
    # main("refcocog", "umd")
    ...
    # peak()
    peak2()
