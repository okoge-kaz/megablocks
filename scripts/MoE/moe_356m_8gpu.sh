#!/bin/bash
#$ -l rt_AF=2
#$ -l h_rt=10:00:00
#$ -j y
#$ -o outputs/MoE/356m_8gpu/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# 512 * 1k * 400k = 200b tokens.
# 512 * 1k * 200k = 100b tokens.
# 512 * 1k * 100k = 50b tokens (default).
# 512 * 1k * 20k = 10b tokens.
TRAINING_STEPS=20000

NUM_EXPERTS=16

CAPACITY_FACTOR=1

TOP_K=1

LOSS_WEIGHT=0.1

BATCH_SIZE=2

##
### Pre-training for MoE 356M parameter.
##

# MoE hyperparameters.
MOE_ARGUMENTS="\
--moe-num-experts=${NUM_EXPERTS} \
--moe-capacity-factor=${CAPACITY_FACTOR} \
--moe-loss-weight=${LOSS_WEIGHT} \
--moe-top-k=${TOP_K}"

# Model hyperparameters.
MODEL_ARGUMENTS="\
--num-layers 24 \
--hidden-size 1024 \
--num-attention-heads 16 \
--seq-length 1024 \
--max-position-embeddings 1024"

# Training hyperparameters.
TRAINING_ARGUMENTS="\
--micro-batch-size ${BATCH_SIZE} \
--global-batch-size 512 \
--train-iters ${TRAINING_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.00015 \
--min-lr 0.00001 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01"

DATASET="datasets/BookCorpusDataset_text_document"

# NOTE: We don't train for enough tokens for the
# split to matter.
DATA_ARGUMENTS="\
--data-path ${DATASET} \
--vocab-file datasets/gpt2-vocab.json \
--merge-file datasets/gpt2-merges.txt \
--make-vocab-size-divisible-by 1024 \
--split 969,30,1"

COMPUTE_ARGUMENTS="\
--fp16 \
--DDP-impl local \
--moe-expert-model-parallelism \
--no-async-tensor-model-parallel-allreduce"

CHECKPOINT_DIR=/groups/gaf51275/llama/checkpoints/MoE/megablocks/moe/356m_8gpu

CHECKPOINT_ARGUMENTS="\
--save-interval 1000 \
--save ${CHECKPOINT_DIR}"

EVALUATION_ARGUMENTS="\
--eval-iters 100 \
--log-interval 1 \
--eval-interval 1000"

# ldconfig
alias ldconfig=/usr/sbin/ldconfig

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -bind-to none -map-by slot \
  -x PATH \
  python third_party/Megatron-LM/pretrain_gpt.py \
  ${MOE_ARGUMENTS} \
  ${MODEL_ARGUMENTS} \
  ${TRAINING_ARGUMENTS} \
  ${DATA_ARGUMENTS} \
  ${COMPUTE_ARGUMENTS} \
  ${CHECKPOINT_ARGUMENTS} \
  ${EVALUATION_ARGUMENTS} \
  --use-mpi \
  --wandb-entity "okoge" \
  --wandb-project "megablock" \
  --wandb-name "MoE_356M_expert=${NUM_EXPERTS}_cap_fac=${CAPACITY_FACTOR}_top_k=${TOP_K}_gb_${BATCH_SIZE}"
