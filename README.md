# SAICR
The code of Symmetric Alignment and Intra-class Contrastive Refinement for Referring Image Segmentation

![the architecture of SAICR](network.png)

# Datasets
Download images from [COCO](https://cocodataset.org/#download). Please use the first downloading link *2014 Train images [83K/13GB]*, and extract the downloaded `train_2014.zip` file.

Data paths should be as follows:

```text
.{YOUR_REFER_PATH}
├── refcoco
├── refcoco+
└── refcocog

.{YOUR_COCO_PATH}
└── train2014
```
