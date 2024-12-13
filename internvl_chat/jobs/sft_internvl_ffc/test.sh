echo "activating env..."

eval "$(/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/chenshaoxiang/workspace/tools/anaconda3/bin/conda shell.bash hook)"
conda activate baichuan

ws_dir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/chenshaoxiang/workspace
lib_dir=${ws_dir}/tools/anaconda3/envs/baichuan/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH=${lib_dir}/cuda_runtime/lib/:${lib_dir}/cudnn/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/chenshaoxiang/workspace/projs/food_ocr/multi-modal-llm/src

wsdir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/chenshaoxiang/workspace/projs/food_ocr/multi-modal-llm
cd ${wsdir}/src

ckpt=${wsdir}/outputs/20231121_111834/2023_11_21-11_24_06-model_MLLM_Baichuan_Vit_Perceiver-lr_5e-05_1e-05-b_1/checkpoints/step_86700.pt

python test/test_mllm_baichuan_official_mmb_highres.py --model_name 20231121_111834 \
    --ckpt $ckpt \
    --use_lora True

