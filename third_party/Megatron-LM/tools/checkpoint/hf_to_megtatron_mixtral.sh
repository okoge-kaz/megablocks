#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -o outputs/
#$ -cwd

set -e

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
cd /bb/llm/gaf51275/llama/taishi-work-streaming/megablocks
source .env/bin/activate


python third_party/Megatron-LM/tools/checkpoint/util.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 1 \
    --load-dir /bb/llm/gaf51275/llm-jp/taishi-work-space/.cache/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/ \
    --save-dir /groups/gaf51275/taishi-work-space/checkpoints/MoE/megablocks/llama2_hf/ \
    --tokenizer-model SentencePieceTokenizer \
    --megatron-path /bb/llm/gaf51275/llama/taishi-work-streaming/megablocks/third_party/Megatron-LM

# python third_party/Megatron-LM/tools/checkpoint/util.py \
#     --model-type GPT \
#     --loader mixtral_hf \
#     --saver megatron \
#     --target-tensor-parallel-size 1 \
#     --load-dir /bb/llm/gaf51275/llm-jp/taishi-work-space/.cache/models--mistralai--Mixtral-8x7B-v0.1/snapshots/58301445dc1378584211722b7ebf8743ec4e192b \
#     --save-dir /groups/gaf51275/taishi-work-space/checkpoints/MoE/megablocks/mixtral/ \
#     --tokenizer-model SentencePieceTokenizer \
#     --megatron-path /bb/llm/gaf51275/llama/taishi-work-streaming/megablocks/third_party/Megatron-LM