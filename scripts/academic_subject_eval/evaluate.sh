#!/bin/bash
set -x
MODEL=${1}
CONFIG=${2}

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
        run.py --config ${CONFIG} --reuse
fi

if [ ${MODEL} == "Qwen2.5-VL-72B" ]; then
    echo "Qwen2.5-VL-72B infer and eval"
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --nproc-per-node=$GPUS \
        --master_port=${MASTER_PORT} \
        run.py --config ${CONFIG} --reuse
fi

if [ ${MODEL} == "InternVL3-8B" ]; then
    echo "InternVL3-8B infer and eval"
    export USE_COT=0
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --nproc-per-node=$GPUS \
        --master_port=${MASTER_PORT} \
        run.py --config ${CONFIG} --reuse
fi

if [[ ${MODEL} == "InternVL3-78B" || ${MODEL} == "InternVL3-38B" ]]; then
    echo "InternVL3-78B or InternVL3-38B infer and eval"
    export USE_COT=0
    export TP=4
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --nproc-per-node=$GPUS \
        --master_port=${MASTER_PORT} \
        run.py --config ${CONFIG} --reuse
fi

if [[ ${MODEL} == "chatgpt-4o-latest" || ${MODEL} == "claude-3-7-sonnet" ]]; then
    echo "${MODEL} infer and eval"
    python run.py --config ${CONFIG} --api-nproc 4 --reuse
fi