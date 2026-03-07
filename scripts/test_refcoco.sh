#!/bin/bash
uname -a
#date
#env
date
#
#85_model_best_refcoco.pth -- testA
#31_model_best_refcoco+.pth -- val
#13_model_best_refcocog_48.pth -- test

# DATA_PATH=./data
DATASET=refcocog
DATA_PATH=/opt/data/share/lsy/zone1/p1/CARIS-main/data
REFER_PATH=/opt/data/share/lsy/zone1/p1/CARIS-main/data
BERT_PATH=pretrained_weights/bert-base-uncased/
MODEL=saicr
SWIN_TYPE=base
IMG_SIZE=448
ROOT_PATH=./output/
RESUME_PATH=${ROOT_PATH}model_refcocog_g.pth
OUTPUT_PATH=${ROOT_PATH}/${DATASET}
SPLIT=val

cd YOUR_CODE_PATH
python eval.py --model ${MODEL} --swin_type ${SWIN_TYPE} \
        --dataset ${DATASET} --split ${SPLIT} \
        --img_size ${IMG_SIZE} --resume ${RESUME_PATH} \
        --bert_tokenizer ${BERT_PATH} --ck_bert ${BERT_PATH} \
        --splitBy google \
        --refer_data_root ${DATA_PATH} --refer_root ${REFER_PATH} 2>&1 | tee ${OUTPUT_PATH}/eval-${SPLIT}.txt


