JOB_ID=20241008_161530
num_gpus=8
num_nodes=2
lr=6e-5
max_seq_length=4096
min_num_frame=90
max_num_frame=120

OUTPUT_DIR='work_dirs/internvl_chat_v2_0/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_ffc_lora'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
JOB_NAME=$(echo $SCRIPTPATH | rev | cut -d '/' -f 1 | rev)

RUN_ARG=$1
if [[ "$RUN_ARG" == "create" ]]; then
    JOB_ID=$(date '+%Y%m%d_%H%M%S')
    echo "Create Mode"
    echo "Script path is $SCRIPTPATH"
    echo "Job name is $JOB_NAME"
    echo "Job ID is $JOB_ID"

    ds_config=../../zero_stage1_config.json
    if [[ -f "$ds_config" ]]; then
        echo "Deepspeed config exists"
        cat $ds_config | grep "train_micro_batch_size_per_gpu"
        cat $ds_config | grep "gradient_accumulation_steps"
        cat $ds_config | grep "lr"
    else
        echo "Deepspeed config not exist"
        exit
    fi

    mkdir -p $JOB_ID
    hope_file=$JOB_ID/${JOB_NAME}_${JOB_ID}.hope
    touch $hope_file
    echo "[roles]" >> $hope_file
    echo "workers = ${num_nodes}" >> $hope_file
    echo "worker.gcores80g = ${num_gpus}" >> $hope_file
    echo "worker.memory = 800000" >> $hope_file
    echo "worker.vcore = 80" >> $hope_file
    echo "worker.ports = 1" >> $hope_file
    echo "worker.script = cat train_multi_node.sh && bash train_multi_node.sh run" >> $hope_file
    cat ../template.hope >> $hope_file
    echo "Created job $hope_file"
    
    echo "JOB_ID=$JOB_ID" > $JOB_ID/train_multi_node.sh
    cat train_multi_node.sh >> $JOB_ID/train_multi_node.sh

    repo_stat=$JOB_ID/stat_${JOB_NAME}_${JOB_ID}.txt
    touch $repo_stat
    echo "-------Current repo status-------" >> $repo_stat
    git status >> $repo_stat
    echo "-------Current changes-------" >> $repo_stat
    git diff >> $repo_stat
    echo "-------Last commit-------" >> $repo_stat
    git show --summary >> $repo_stat

    run_script=$JOB_ID/run.sh
    echo "hope run ${JOB_NAME}_${JOB_ID}.hope | tee -a hope_submit.log" >> $run_script

    exit
elif [[ "$RUN_ARG" == "run" ]]; then
    echo "Run Mode"
else
    echo "invalid RUN_ARG"
    exit
fi

eval "$(/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/gaolishuai/anaconda3/bin/conda shell.bash hook)"
conda activate internvl
echo "env activated"

ifconfig
nvidia-smi

wsdir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/gaolishuai/project/research
proj_dir=${wsdir}/InternVL
misc_dir=${wsdir}/misc
# prepare ssh files
mkdir -p ~/.ssh/
cp -r ${misc_dir}/ssh/* ~/.ssh/
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
ls -al ~/
ls -al ~/.ssh/

which deepspeed

cd ${proj_dir} || exit
echo "-------Current repo status-------"
git status
echo "-------Current changes-------"
git diff
echo "-------Last commit-------"
git show --summary
cd internvl_chat || exit

cd jobs || exit
cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
echo "cluster spec is $cluster_spec"
worker_list_command="import json_parser;print(json_parser.parse(\"$cluster_spec\", \"worker\"))"
echo "worker list command is $worker_list_command"
eval worker_list=`python -c "$worker_list_command"`
worker_strs=(${worker_list//,/ })
master=${worker_strs[0]}
echo "master is $master"
master_strs=(${master//:/ })
master_addr=${master_strs[0]}
master_port=${master_strs[1]}
echo "master address is $master_addr"
echo "master port is $master_port"
dist_url="tcp://$master_addr:$master_port"

index_command="import json_parser;print(json_parser.parse(\"$cluster_spec\", \"index\"))"
eval node_rank=`python -c "$index_command"`
echo "node rank is $node_rank"

master_node=${worker_strs[0]}
echo "master node is $master_node"

flag_filedir=${misc_dir}/job-finished-flag
mkdir -p $flag_filedir
flag_filepath=$flag_filedir/$master_node
echo "flag filepath is $flag_filepath"

export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export RANK=$node_rank

if (($node_rank == 0))
then
    python generate_hostfile.py
    echo "Show content in /workdir/TMPHOSTFILE :"
    cat /workdir/TMPHOSTFILE
    
    HOST_FILE_PATH="/workdir/TMPHOSTFILE"
    echo $HOST_FILE_PATH
    
    wait_cnt=0
    while [ $wait_cnt -lt 720 ];
    do
        all_worker_ready=true
        for (( i=1; i<${#worker_strs[@]}; i++ ));
        do
            if [ ! -f "$flag_filedir/${worker_strs[$i]}" ];
            then
                echo "worker $i is not ready!"
                all_worker_ready=false
                break
            fi
        done
        if [ $all_worker_ready = true ];
        then
            echo "All workers have been ready! start train ..."
            break
        else
            echo "Waiting other workers to be ready ..."
            sleep 5
        fi
        wait_cnt=$(($wait_cnt+1))
    done

    sleep 10

    cd ..
    NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 deepspeed \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=${GPUS} \
        --master_port=$MASTER_PORT
        internvl/train/internvl_chat_finetune.py \
        --model_name_or_path "./pretrained/InternVL2-1B" \
        --conv_style "Hermes-2" \
        --output_dir ${OUTPUT_DIR} \
        --meta_path "./shell/data/internvl_1_2_finetune_custom_video.json" \
        --overwrite_output_dir True \
        --use_video_frames_compress True \
        --min_num_frame ${min_num_frame} \
        --max_num_frame ${max_num_frame} \
        --num_video_query_token 16 \
        --force_image_size 448 \
        --max_dynamic_patch 6 \
        --down_sample_ratio 0.5 \
        --drop_path_rate 0.0 \
        --freeze_llm True \
        --freeze_mlp True \
        --freeze_backbone True \
        --use_llm_lora 16 \
        --vision_select_layer -1 \
        --dataloader_num_workers 4 \
        --bf16 True \
        --num_train_epochs 1 \
        --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACC} \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 200 \
        --save_total_limit 1 \
        --learning_rate ${lr} \
        --weight_decay 0.01 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --max_seq_length ${max_seq_length} \
        --do_train True \
        --grad_checkpoint True \
        --group_by_length True \
        --dynamic_image_size True \
        --use_thumbnail True \
        --ps_version 'v2' \
        --deepspeed "zero_stage1_config.json" \
        --report_to "tensorboard" \
        2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
        
    touch $flag_filepath
    echo "master job finished, generate flag file $flag_filepath"
else
    node_ready_flag_filepath=$flag_filedir/${worker_strs[$node_rank]}
    sleep 1m
    touch $node_ready_flag_filepath
    echo "node $node_rank is ready, generate ready flag file $node_ready_flag_filepath"
    while [ ! -f $flag_filepath ]
    do
        echo "master job doesnot finished, continue sleep..."
        sleep 5
    done
fi