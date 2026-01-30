export CUDA_VISIBLE_DEVICES=0,2,4,5
export RAY_TEMP_DIR="/root/dws/3D_QA/ray_temp"

export HF_HUB_OFFLINE=1
export CURL_CA_BUNDLE=''

rm -rf /root/dws/3D_QA/ray_temp/*

python3 /root/dws/3D_QA/geo/ScanQA_SQA/cdviews/retrieval_base_scanqa_multi.py \
    --model_name qwen \
    --llava_model-path /root/dws/3D_QA/TStar/cdViews/model/llava-onevision-qwen2-7b-ov \
    --qwen_model-path /root/dws/MCS/Models/Qwen2.5-VL-7B-Instruct \
    --llava_video_model_path /root/dws/3D_QA/TStar/cdViews/model/LLaVA-Video-7B-Qwen2\
    --cfg_file /root/dws/3D_QA/TStar/cdViews/cfgs/QA.yaml \
    --mllm_device cuda \
    --blip_device cuda \
    --num_frames 16\
    --pure_retrieval \
    --sample_strategy aks \
    --method_type "qwen_aks_16" \
    --dataset ScanQA \
