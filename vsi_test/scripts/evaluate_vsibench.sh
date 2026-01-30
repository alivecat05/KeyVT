#!/bin/bash
export HF_HUB_OFFLINE=1
export CURL_CA_BUNDLE=''
cd "$(dirname "$0")"
cd ..
export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


export QWEN_IS_COMPRESS=True
export QWEN_COMPRESS_RATIO=0.5
export QWEN_IS_DIVPRUNE=False
export QWEN_IS_OT=False
python evaluate/eval_vsibench2dvl.py \
    --model_name qwen\
    --batch_size 1 \
    --nframes 8\
    --sample_strategy space_aks \
    --is_compress