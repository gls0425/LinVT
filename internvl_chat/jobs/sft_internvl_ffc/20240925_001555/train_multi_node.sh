JOB_ID=20231216_001555
num_gpus=8
num_nodes=8
bs=1
gacc=1
lr=2e-5
lr_min=2e-6
warmup=500

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
JOB_NAME=$(echo $SCRIPTPATH | rev | cut -d '/' -f 1 | rev)

RUN_ARG=$1
if [[ "$RUN_ARG" == "create" ]]; then
    JOB_ID=$(date '+%Y%m%d_%H%M%S')
    echo "Create Mode"
    echo "Script path is $SCRIPTPATH"
    echo "Job name is $JOB_NAME"
    echo "Job ID is $JOB_ID"

    ds_config=../../ds_configs/ds_config_1_bs${bs}_acc${gacc}_${lr}.json
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

eval "$(/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/chenshaoxiang/workspace/tools/anaconda3/bin/conda shell.bash hook)"
conda activate baichuan
echo "env activated"

ifconfig
nvidia-smi

wsdir=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/chenshaoxiang/workspace/dev
proj_dir=${wsdir}/normal_dataloader_optimized
misc_dir=${wsdir}/misc
out_dir=${wsdir}/outputs/${JOB_ID}
mkdir -p ${out_dir}


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

cd ${proj_dir}
echo "-------Current repo status-------"
git status
echo "-------Current changes-------"
git diff
echo "-------Last commit-------"
git show --summary

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

cd src

echo "Call the load model and tokenizer code.........................."
python util/load_model_tokenizer.py
echo "[Done] Call the load model and tokenizer code.........................."

cd ..

if (($node_rank == 0))
then
    python generate_hostfile.py
    echo "Show content in /workdir/TMPHOSTFILE :"
    cat /workdir/TMPHOSTFILE
    
    HOST_FILE_PATH="/workdir/TMPHOSTFILE"
    echo $HOST_FILE_PATH

    cd ./src
    
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

    NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 deepspeed \
        --master_port $MASTER_PORT --num_nodes ${num_nodes} --num_gpus ${num_gpus} --hostfile $HOST_FILE_PATH main_e2e_deepspeed.py \
        --save-frequency 1 \
        --logs ${out_dir} \
        --report-to tensorboard \
        --warmup=${warmup} \
        --batch-size=${bs} \
        --lr=${lr} \
        --lr_min=${lr_min} \
        --wd=0.05 \
        --stage=sft \
        --workers=2 \
        --multi_round_loss \
        --model MLLM_Baichuan_Vit_Perceiver \
        --deepspeed --deepspeed_config ../ds_configs/ds_config_1_bs${bs}_acc${gacc}_${lr}.json \
        --pretrain /mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/chenshaoxiang/workspace/dev/outputs/20231208_001434/2023_12_13-00_12_40-model_MLLM_Baichuan_Vit_Perceiver-lr_5e-05_1e-05-b_2/checkpoints/epoch_20.pt \
        --use_lora \
        --high_res \
        --lora_rank 128 \
        --lora_alpha 256 \
        --vit_trainable attn_pool,ln_post,proj,transformer.resblocks.47,transformer.resblocks.46,transformer.resblocks.45,transformer.resblocks.44,transformer.resblocks.43,transformer.resblocks.42,transformer.resblocks.41 \
        --data_configs en_m3it_2m,en_sharegpt4v_666k,en_refcoco@rec_330k,en_refcoco@reg_330k,zh_llava_350k,zh_synthdog@col3_600k,en_synthdog@col3_300k,zh_synthdogqa_600k,en_synthdogqa_300k,zh_pdfsumm_36k \
        --num_splits=1 \
        --epochs=1
        
    touch $flag_filepath
    echo "master job finished, generate flag file $flag_filepath"
else
    node_ready_flag_filepath=$flag_filedir/${worker_strs[$node_rank]}
    sleep 1m
    touch $node_ready_flag_filepath
    echo "node $node_rank is ready, generate ready flag file $node_ready_flag_filepath"
    cd ./src
    while [ ! -f $flag_filepath ]
    do
        echo "master job doesnot finished, continue sleep..."
        sleep 5
    done
fi
