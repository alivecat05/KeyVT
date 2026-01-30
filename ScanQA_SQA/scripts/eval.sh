export CUDA_VISIBLE_DEVICES=0,2,4,5

export HF_HUB_OFFLINE=1
export CURL_CA_BUNDLE=''

rm -rf ./3D_QA/ray_temp/*

python3 ./geo/ScanQA_SQA/cdviews/retrieval_base_scanqa_multi.py \
    --model_name qwen \
    --mllm_device cuda \
    --blip_device cuda \
    --num_frames 16\
    --pure_retrieval \
    --sample_strategy aks \
    --dataset ScanQA \
