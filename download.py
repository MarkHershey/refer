# https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip
# https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
# https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
# https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip


import os
import subprocess

URLS = [
    # "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip",
    "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
    "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
    "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
]

DATA_DIR = "data"


def main():
    for url in URLS:
        subprocess.run(["wget", url, "-P", DATA_DIR])
        subprocess.run(
            ["unzip", os.path.join(DATA_DIR, url.split("/")[-1]), "-d", DATA_DIR]
        )
        os.remove(os.path.join(DATA_DIR, url.split("/")[-1]))


if __name__ == "__main__":
    main()
