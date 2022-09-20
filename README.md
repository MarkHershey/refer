# Referring Expression Comprehension/Segmentation Data API

## Dependencies

-   Python 3.X

```bash
pip install -r requirements.txt
```

## Download Data

```bash
python3 download.py
```

## Example

### `"data/refcoco/instances.json"` Top-level keys:

-   `info`
-   `images`
-   `licenses`
-   `annotations`
-   `categories`

### `data["images"][0]` in `"data/refcoco/instances.json"`

```json
{
    "license": 1,
    "file_name": "COCO_train2014_000000098304.jpg",
    "coco_url": "http://mscoco.org/images/98304",
    "height": 424,
    "width": 640,
    "date_captured": "2013-11-21 23:06:41",
    "flickr_url": "http://farm6.staticflickr.com/5062/5896644212_a326e96ea9_z.jpg",
    "id": 98304
}
```

### `data["annotations"][0]` in `"data/refcoco/instances.json"`

```json
{
    "segmentation": [
        [
            267.52, 229.75, 265.6, 226.68, 265.79, 223.6, 263.87, 220.15,
            263.87, 216.88, 266.94, 217.07, 268.48, 221.3, 272.32, 219.95,
            276.35, 220.15, 279.62, 218.03, 283.46, 218.42, 285.0, 220.92,
            285.0, 223.22, 284.42, 224.95, 280.96, 225.14, 279.81, 226.48,
            281.73, 228.41, 279.43, 229.37, 275.78, 229.17, 273.86, 229.56,
            274.24, 232.05, 269.82, 231.67, 267.14, 231.48, 266.75, 228.6
        ]
    ],
    "area": 197.29899999999986,
    "iscrowd": 0,
    "image_id": 98304,
    "bbox": [263.87, 216.88, 21.13, 15.17],
    "category_id": 18,
    "id": 3007
}
```

### A `ref` object stored in `refs(dataset).p`

```json
{
    "sent_ids": [0, 1, 2],
    "file_name": "COCO_train2014_000000581857_16.jpg", // ignore this
    "ann_id": 1719310,
    "ref_id": 0,
    "image_id": 581857,
    "split": "train",
    "sentences": [
        {
            "tokens": ["the", "lady", "with", "the", "blue", "shirt"],
            "raw": "THE LADY WITH THE BLUE SHIRT",
            "sent_id": 0,
            "sent": "the lady with the blue shirt"
        },
        {
            "tokens": ["lady", "with", "back", "to", "us"],
            "raw": "lady w back to us",
            "sent_id": 1,
            "sent": "lady with back to us"
        },
        {
            "tokens": ["blue", "shirt"],
            "raw": "blue shirt",
            "sent_id": 2,
            "sent": "blue shirt"
        }
    ],
    "category_id": 1
}
```

## Prepare Images:

Besides, add "mscoco" into the `data/images` folder, which can be from [mscoco](http://mscoco.org/dataset/#overview)
COCO's images are used for RefCOCO, RefCOCO+ and refCOCOg.
For RefCLEF, please add `saiapr_tc-12` into `data/images` folder. We extracted the related 19997 images to our cleaned RefCLEF dataset, which is a subset of the original [imageCLEF](http://imageclef.org/SIAPRdata). Download the [subset](https://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip) and unzip it to `data/images/saiapr_tc-12`.

## How to use

The "refer.py" is able to load all 4 datasets with different kinds of data split by UNC, Google, UMD and UC Berkeley.
**Note for RefCOCOg, we suggest use UMD's split which has train/val/test splits and there is no overlap of images between different split.**

```python
# locate your own data_root, and choose the dataset_splitBy you want to use
# refer = REFER(data_root, dataset='refclef',  splitBy='unc')
# refer = REFER(data_root, dataset='refclef',  splitBy='berkeley') # 2 train and 1 test images missed
refer = REFER(data_root, dataset='refcoco',  splitBy='unc')
# refer = REFER(data_root, dataset='refcoco',  splitBy='google')
refer = REFER(data_root, dataset='refcoco+', splitBy='unc')
refer = REFER(data_root, dataset='refcocog', splitBy='umd')      # Recommended, including train/val/test
refer = REFER(data_root, dataset='refcocog', splitBy='google')   # test split not released yet
```

## Notes

important modifications commented as `NOTE: added by Mark`
