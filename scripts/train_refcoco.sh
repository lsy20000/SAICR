#!/bin/bash
uname -a
#date
#env
date
# pkill -4 python
# --resume ./output/mychange/refcocog/2_model_best_refcocog_40.pth
# --resume ./output/mychange/refcoco+/24_4_model_best_refcoco+.pth
DATASET=refcoco+
DATA_PATH=/opt/data/share/lsy/zone1/p1/CARIS-main/data
REFER_PATH=/opt/data/share/lsy/zone1/p1/CARIS-main/data
MODEL=saicr
SWIN_PATH=./pretrained_weights/swin_base_patch4_window7_224_22k.pth
BERT_PATH=./pretrained_weights/bert-base-uncased/
OUTPUT_PATH=./output/refcoco+
IMG_SIZE=448 
now=$(date +"%Y%m%d_%H%M%S")

mkdir ${OUTPUT_PATH}
mkdir ${OUTPUT_PATH}/${DATASET}

cd YOUR_CODE_PATH
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py --model ${MODEL} \
        --dataset ${DATASET} --model_id ${DATASET} --batch-size 8 --pin_mem --print-freq 100 --workers 8 \
        --lr 1e-4 --wd 1e-2 --swin_type base --splitBy unc \
        --warmup --warmup_ratio 1e-3 --warmup_iters 1500 --clip_grads --clip_value 1.0 \
        --pretrained_swin_weights ${SWIN_PATH} --epochs 50 --img_size ${IMG_SIZE} \
        --bert_tokenizer ${BERT_PATH} --ck_bert ${BERT_PATH} --output-dir ${OUTPUT_PATH} \
        --refer_data_root ${DATA_PATH} --refer_root ${REFER_PATH} 2>&1 | tee ${OUTPUT_PATH}'/'${DATASET}'/'train-${now}.txt &
