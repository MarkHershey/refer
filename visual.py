import csv
from email import header
from pathlib import Path

import matplotlib.pyplot as plt

from refer import REFER

DATA_DIR = Path("__file__").resolve().parent / "data"


def main(dataset: str, splitBy: str, ref_id: int, save_dir: str = "nv-tmp"):
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
    return str(save_dir / f"{dataset}_{splitBy}_ref{ref_id}.png")


def find_ref_id(dataset="refcoco+", splitBy="unc", setName="train", idx: int = None):
    """
    Note that this func assumes that idx supplied is after offset
    """
    custom_dir = Path("/home/markhh/CODE/DEEP_LEARNING/novel_composition/data/custom")
    assert custom_dir.exists()
    custom_file = custom_dir / f"{dataset}_{splitBy}_{setName}.csv"
    assert custom_file.exists(), f"{custom_file}"

    with open(custom_file, "r") as f:
        csv_reader = list(csv.reader(f))
        header = csv_reader.pop(0)
        for row in csv_reader:
            if int(row[0]) == idx:
                return int(row[2])

    raise ValueError(f"Cannot find {idx} in {custom_file}")


def get(save_dir="nv10", dataset="refcoco+", setName="train", idx: int = None):
    return main(
        dataset=dataset,
        splitBy="umd" if dataset == "refcocog" else "unc",
        save_dir=save_dir,
        ref_id=find_ref_id(
            dataset=dataset,
            splitBy="umd" if dataset == "refcocog" else "unc",
            setName=setName,
            idx=idx,
        ),
    )


def find_orange_juice():
    fp = "/home/mark/code/novel_composition/data/custom/refcoco+_unc_train.csv"
    idxes = []
    with open(fp, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "orange juice" in line:
                print(line)
                idxes.append(int(line.split(",")[0]))

    for i in idxes:
        print(i)
        get("nv-tmp-orange", "refcoco+", "train", i)


def find():
    import json

    fp = "../DEEP_LEARNING/novel_composition/data/c_finetune_refcoco_train.json"
    with open(fp, "r") as f:
        data = json.load(f)

    sents = []
    pics = []

    for idx, sent in data.items():
        tokens = sent.split()
        if len(tokens) > 4:
            continue

        if "coffee" not in tokens:
            continue

        sents.append(sent)
        pic = get("p1-3", "refcoco", "train", int(idx))
        pics.append(pic)

    md_lines = ["| sent | pic |", "| --- | --- |"]
    for sent, pic in zip(sents, pics):
        md_lines.append(f"| {sent} | ![]({pic}) |")

    with open("coffee.md", "w") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    ...
    # main("refcoco+", "unc", ref_id=861)
    # main("refcoco+", "unc", ref_id=19794)
    # main("refcoco+", "unc", ref_id=48870)
    # main("refcoco+", "unc", ref_id=839)
    # main("refcoco", "unc", ref_id=26910)

    # get("nv-tmp", "refcoco+", "val", 252014) # 252014 [person] [against wall]
    # get("nv-tmp", "refcoco+", "val", 253890) # 253890 [jumping] [on skate board]
    # get("nv-tmp", "refcoco+", "val", 253988) # 253988 [motorcycle] [in the foreground]
    # get("nv-tmp", "refcoco+", "val", 258726) # 258726 [brown doughnut] [no sprinkles]

    # get("nv-pp", "refcoco", "testB", 9617) # 9617 [white bear] [in red sweater]
    # get("nv-pp", "refcoco", "train", 9460) # "9460": "left white bear",
    # get("nv-pp", "refcoco", "train", 26934) # "26934": "man in red sweater",

    # get("nv-wp", "refcocog", "test", 7527) # 7527 [lighting] [a birthday cake]
    # get("nv-wp", "refcocog", "train", 303023) # "303023": "a man in a coat lighting a cigarette"
    # get("nv-wp", "refcocog", "train", 324916) # "324916": "a birthday cake on a purple table",

    # get("nv-wp2", "refcoco+", "val", 253988) # 253988 [motorcycle] [in the foreground]
    # get("nv-wp2", "refcoco+", "train", 174869) # "174869": "blue and white motorcycle",
    # get("nv-wp2", "refcoco+", "train", 170692) # "170692": "gray couch in the foreground",

    # get("nv-wp3", "refcoco+", "val", 252014) # 252014 [person] [against wall]
    # get("nv-wp3", "refcoco+", "train", 185498) # "185498": "screens against wall",
    # get("nv-wp3", "refcoco+", "train", 185932) # "185932": "person jumping high into air",
    # get("nv-wp3", "refcoco+", "train", 206601) # "206601": "empty seat section against wall"

    # get("nv-ww", "refcoco+", "val", 258020) # 258020 [wooden] [seat]
    # get("nv-ww", "refcoco+", "train", 246961) # "246961": "wooden table",

    # get("p1", "refcoco", "train", 74271)
    # get("p1", "refcoco", "train", 26756)
    # get("p1", "refcoco", "train", 8134)
    # get("p1", "refcoco", "train", 60756)

    # get("p1", "refcoco", "train", 90922) # ref 37660
    # get("p1", "refcoco", "train", 12363) # ref 5080
    # get("p1", "refcoco", "train", 91197)  # ref 37774

    # get("p1", "refcoco", "train", 33872)  # ref 14015
    # get("p1", "refcoco", "train", 13826)  # ref 5696

    # get("p1-1", "refcoco", "train", 4111)  # ref
    # get("p1-1", "refcoco", "train", 7241)  # ref
    # get("p1-1", "refcoco", "train", 87061)  # ref
    find()
