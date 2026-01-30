cd "$(dirname "$0")"
cd ..


export CUDA_VISIBLE_DEVICES=2,4,5


python evaluate/init_detecor.py