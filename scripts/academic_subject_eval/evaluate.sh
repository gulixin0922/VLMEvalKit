#!/bin/bash
set -x
MODEL=${1}

MASTER_PORT=${MASTER_PORT:-63669}
GPUS=${GPUS:-8}

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      GPUS=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
echo "GPUS: ${GPUS}"

if [ ${MODEL} == "Qwen2.5-VL-7B" ]; then
    echo "Qwen2.5-VL-7B infer and eval"
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --nproc-per-node=$GPUS \
        --master_port=${MASTER_PORT} \
        run.py --config scripts/academic_subject_eval/qwen2_5_vl_7b_config.json --reuse
fi

if [ ${MODEL} == "Qwen2.5-VL-72B" ]; then
    echo "Qwen2.5-VL-72B infer and eval"
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --nproc-per-node=$GPUS \
        --master_port=${MASTER_PORT} \
        run.py --config scripts/academic_subject_eval/qwen2_5_vl_72b_config.json --reuse
fi

if [ ${MODEL} == "InternVL3-78B" ]; then
    echo "InternVL3-78B infer and eval"
    export USE_COT=0
    export TP=4
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --nproc-per-node=$GPUS \
        --master_port=${MASTER_PORT} \
        run.py --config scripts/academic_subject_eval/internvl3_78b_config.json --reuse
fi

if [ ${MODEL} == "chatgpt-4o-latest" ]; then
    echo "chatgpt-4o-latest infer and eval"
    python run.py --config scripts/academic_subject_eval/openai_config.json --api-nproc 4 --reuse
fi

if [ ${MODEL} == "claude-3-7-sonnet" ]; then
    echo "claude-3-7-sonnet infer and eval"
    python run.py --config scripts/academic_subject_eval/claude_config.json --api-nproc 4 --reuse
fi