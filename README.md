# SAICR
The code of Symmetric Alignment and Intra-class Contrastive Refinement for Referring Image Segmentation

<img src="network.png" alt="the architecture of SAICR" width="70%">

Referring Image Segmentation aims to precisely segment a specific object within an image based on a natural language description. However, existing methods often suffer from imprecise visual-language alignment due to suboptimal cross-modal interactions and struggle to distinguish similar distractors because of insufficient feature discriminability. To address these challenges, we propose the Symmetric Alignment and Intra-class Contrastive Refinement (SAICR) framework. For more precise alignment, we achieve multi-level feature fusion in the encoder stage through a bidirectional interaction module, composed of a Symmetric Cross-Attention Fusion (SCAF) module and a Spatial-Channel Gating (SCG) module. For discriminating intra-class objects, we introduce a Dual Contrastive Feature Refinement (DCFR) module, which is the first to apply the idea of Contrastive Clustering to this task, enhancing the model’s discriminative power by combining instance-level and cluster-level learning. Extensive experiments demonstrate that SAICR outperforms existing SOTA methods on three public benchmarks.

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

# Pretrained Models

Download pretrained [Swin-B](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) and [BERT-B](https://huggingface.co/bert-base-uncased/tree/main). 

# Code

Our models will be released soon.
