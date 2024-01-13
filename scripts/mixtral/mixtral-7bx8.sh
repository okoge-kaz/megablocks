#!/bin/bash
#$ -l rt_AF=16
#$ -l h_rt=10:00:00
#$ -j y
#$ -o outputs/mixtral-7bx8/
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

# training settings
TRAIN_ITERATIONS=63312

# MoE hyperparameters.
NUM_EXPERTS=8
CAPACITY_FACTOR=0
TOP_K=2
LOSS_WEIGHT=0.1

# Pre-training for Mixtral 7Bx8
# https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
NUM_LAYERS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336
NUM_ATTENTION_HEADS=32
NUM_KEY_VALUE_HEADS=8

LR=1.0e-4
MIN_LR=1.0e-6
INIT_STD=0.02

SEQUENCE_LENGTH=4096

WEIGHT_DECAY=0.1

# Training hyperparameters.
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024

# data config
TOKENIZER_MODEL=/bb/llm/gaf51275/llm-jp/llm-ja-tokenizer/models/ver2/code10K_en20K_ja30K.ver2.2.model
DATASET_DIR=/bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/train

TRAIN_DATA_PATH=""

# ja wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 1489457253 ${DATASET_DIR}/ja_wiki/ja_wiki_merge_1_text_document"
# en wiki
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 4983898399 ${DATASET_DIR}/en_wiki/en_wiki_merge_1_text_document"
# code stack
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8967214774 ${DATASET_DIR}/code_stack/code_stack_merge_1_text_document"
# en pile
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17716652494 ${DATASET_DIR}/en_pile/en_pile_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17728398911 ${DATASET_DIR}/en_pile/en_pile_merge_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17862741217 ${DATASET_DIR}/en_pile/en_pile_merge_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17854181202 ${DATASET_DIR}/en_pile/en_pile_merge_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17779824310 ${DATASET_DIR}/en_pile/en_pile_merge_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17847796716 ${DATASET_DIR}/en_pile/en_pile_merge_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8938950206 ${DATASET_DIR}/en_pile/en_pile_merge_7_text_document"
# ja cc
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19540410239 ${DATASET_DIR}/ja_cc/ja_cc_merge_1_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19559059958 ${DATASET_DIR}/ja_cc/ja_cc_merge_2_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19547251566 ${DATASET_DIR}/ja_cc/ja_cc_merge_3_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19550089401 ${DATASET_DIR}/ja_cc/ja_cc_merge_4_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19553509796 ${DATASET_DIR}/ja_cc/ja_cc_merge_5_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 19566479585 ${DATASET_DIR}/ja_cc/ja_cc_merge_6_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 17060823775 ${DATASET_DIR}/ja_cc/ja_cc_merge_7_text_document"

# validation data
VALIDATION_DATASET_PATH="/bb/llm/gaf51275/llm-jp/binarize/gpt-7b/ver2.2/code10K_en20K_ja30K/val"

VALIDATION_DATA_PATH=""
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 77810430 ${VALIDATION_DATASET_PATH}/code_stack_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 37133061 ${VALIDATION_DATASET_PATH}/en_pile_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 1011609 ${VALIDATION_DATASET_PATH}/en_wiki_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 147265562 ${VALIDATION_DATASET_PATH}/ja_cc_validation_0_text_document"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH} 1097003 ${VALIDATION_DATASET_PATH}/ja_wiki_validation_0_text_document"

# checkpoint settings
CHECKPOINT_DIR=/groups/gaf51275/llama/checkpoints/MoE/megablocks/moe/mixtral-7bx8_${NUM_EXPERTS}expert_${CAPACITY_FACTOR}cap_fac_${TOP_K}top_k_${BATCH_SIZE}gb/

mkdir -p ${CHECKPOINT_DIR}

# ldconfig
alias ldconfig=/usr/sbin/ldconfig

# distributed settings
TENSER_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=4

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
  --moe-num-experts=${NUM_EXPERTS} \
  --moe-capacity-factor=${CAPACITY_FACTOR} \
  --moe-loss-weight=${LOSS_WEIGHT} \
  --moe-top-k=${TOP_K} \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_ATTENTION_HEADS} \
  --seq-length ${SEQUENCE_LENGTH} \
  --max-position-embeddings ${SEQUENCE_LENGTH} \
  --micro-batch-size ${BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_ITERATIONS} \
  --lr-decay-iters ${TRAIN_ITERATIONS} \
  --init-method-std ${INIT_STD} \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-warmup-fraction 0.01 \
  --clip-grad 1.0 \
  --weight-decay ${WEIGHT_DECAY} \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --train-data-path ${TRAIN_DATA_PATH} \
  --valid-data-path ${VALIDATION_DATA_PATH} \
  --distributed-backend nccl \
  --bf16 \
  --DDP-impl local \
  --tensor-model-parallel-size ${TENSER_MODEL_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLEL_SIZE} \
  --moe-expert-model-parallelism \
  --no-async-tensor-model-parallel-allreduce \
  --save-interval 500 \
  --save ${CHECKPOINT_DIR} \
  --load ${CHECKPOINT_DIR} \
  --eval-iters 5 \
  --log-interval 1 \
  --eval-interval 5 \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity full \
  --no-bias-gelu-fusion \
  --use-mpi \
  --wandb-entity "okoge" \
  --wandb-project "megablock" \
  --wandb-name "Mixtral-7Bx8_expert=${NUM_EXPERTS}_cap_fac=${CAPACITY_FACTOR}_top_k=${TOP_K}_gb_${BATCH_SIZE}"

# normalization
# recompute-gradularity
# no-bias
# swiglu
# rotary positoinal embedding
# untie embedding
# attention dropout
# hidden dropout
