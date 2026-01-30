#!/bin/bash
export HF_HUB_OFFLINE=1
export CURL_CA_BUNDLE=''
# change to workspace root directory
cd "$(dirname "$0")"
cd ..
export RAY_TEMP_DIR="/root/dws/3D_QA/ray_temp"
export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7



##########################################&&777777777##############################


# export QWEN_IS_COMPRESS=True
# export QWEN_COMPRESS_RATIO=0.5
# export QWEN_IS_DIVPRUNE=False
# export QWEN_IS_OT=False
# rm -rf /root/dws/3D_QA/ray_temp/*
# python evaluate/eval_vsibench2dvl.py \
#     --model_name qwen\
#     --video_root /root/dws/3D_QA/Spatial-MLLM-master/evaluate/annotation/VSIBench \
#     --model_type qwen7b_floc_noprom\
#     --llava_video_model_path /root/dws/3D_QA/TStar/cdViews/model/LLaVA-Video-7B-Qwen2 \
#     --qwen_model_path /root/dws/MCS/Models/Qwen2.5-VL-7B-Instruct \
#     --llava_model-path /root/dws/3D_QA/TStar/cdViews/model/llava-onevision-qwen2-0.5b-ov \
#     --batch_size 1 \
#     --nframes 8\
#     --sample_strategy space_aks \
#     --is_compress



# rm -rf /root/dws/3D_QA/ray_temp/*
# python evaluate/eval_vsibench2dvl.py \
#     --model_name llavavideo\
#     --video_root /root/dws/3D_QA/Spatial-MLLM-master/evaluate/annotation/VSIBench \
#     --model_type llavavideo_aks_8\
#     --llava_video_model_path /root/dws/3D_QA/TStar/cdViews/model/LLaVA-Video-7B-Qwen2 \
#     --qwen_model_path /root/dws/3D_QA/TStar/cdViews/model/Qwen2.5_vl_instruct_3B \
#     --llava_model-path /root/dws/3D_QA/TStar/cdViews/model/llava-onevision-qwen2-7b-ov \
#     --batch_size 1 \
#     --nframes 8\
#     --sample_strategy aks \

# export OT_UPDATE_STEP=15

# export OT_TOPK=20
# rm -rf /root/dws/3D_QA/ray_temp/*
# python evaluate/eval_vsibench2dvl.py \
#     --model_name llavavideo\
#     --video_root /root/dws/3D_QA/Spatial-MLLM-master/evaluate/annotation/VSIBench \
#     --model_type llavavideo_keyvt_16_U5T10\
#     --llava_video_model_path /root/dws/3D_QA/TStar/cdViews/model/LLaVA-Video-7B-Qwen2 \
#     --qwen_model_path /root/dws/3D_QA/TStar/cdViews/model/Qwen2.5_vl_instruct_3B \
#     --llava_model-path /root/dws/3D_QA/TStar/cdViews/model/llava-onevision-qwen2-7b-ov \
#     --batch_size 1 \
#     --nframes 16\
#     --sample_strategy space_aks \
#     --is_compress \
#     --ratio 0.5 \
#     --is_ot \



rm -rf /root/dws/3D_QA/ray_temp/*
python evaluate/eval_vsibench2dvl.py \
    --model_name llavavideo\
    --video_root /root/dws/3D_QA/Spatial-MLLM-master/evaluate/annotation/VSIBench \
    --model_type llavavideo_keyvt_16_uniform\
    --llava_video_model_path /root/dws/3D_QA/TStar/cdViews/model/LLaVA-Video-7B-Qwen2 \
    --qwen_model_path /root/dws/3D_QA/TStar/cdViews/model/Qwen2.5_vl_instruct_3B \
    --llava_model-path /root/dws/3D_QA/TStar/cdViews/model/llava-onevision-qwen2-7b-ov \
    --batch_size 1 \
    --nframes 8\
    --sample_strategy uniform \





# rm -rf /root/dws/3D_QA/ray_temp/*
# python evaluate/eval_vsibench2dvl.py \
#     --model_name llava\
#     --video_root /root/dws/3D_QA/Spatial-MLLM-master/evaluate/annotation/VSIBench \
#     --model_type llava7b_floc_t16\
#     --llava_video_model_path /root/dws/3D_QA/TStar/cdViews/model/LLaVA-Video-7B-Qwen2 \
#     --qwen_model_path /root/dws/3D_QA/TStar/cdViews/model/Qwen2.5_vl_instruct_3B \
#     --llava_model-path /root/dws/3D_QA/TStar/cdViews/model/llava-onevision-qwen2-7b-ov \
#     --batch_size 1 \
#     --nframes 8\
#     --sample_strategy space_aks \
#     --is_compress \
#     --ratio 0.5 \

