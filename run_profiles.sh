#!/usr/bin/env bash
set -e

CTX=(128 256 512 1024)

declare -A DMODEL=(
  [small]=768
  [medium]=1024
  [large]=1280
  [xl]=1600
  [2p7b]=2560
)

declare -A DFF=(
  [small]=3072
  [medium]=4096
  [large]=5120
  [xl]=6400
  [2p7b]=10240
)

declare -A LAYERS=(
  [small]=12
  [medium]=24
  [large]=36
  [xl]=48
  [2p7b]=32
)

declare -A HEADS=(
  [small]=12
  [medium]=16
  [large]=20
  [xl]=25
  [2p7b]=32
)

for SIZE in small medium large xl 2p7b; do
  for C in "${CTX[@]}"; do

    OUT="nsys_${SIZE}_ctx${C}"

    echo "=================================================="
    echo "Profiling: $SIZE | context = $C"
    echo "Output: ${OUT}.nsys-rep"
    echo "=================================================="

    set +e
    uv run nsys profile \
      -o ${OUT} \
      --force-overwrite true \
      --python-backtrace=cuda \
      --pytorch=autograd-nvtx \
      python cs336_systems/benchmark.py \
        --device cuda \
        --backward \
        --context_length ${C} \
        --d_model ${DMODEL[$SIZE]} \
        --d_ff ${DFF[$SIZE]} \
        --num_layers ${LAYERS[$SIZE]} \
        --num_heads ${HEADS[$SIZE]}
    STATUS=$?
    set -e

    if [ $STATUS -ne 0 ]; then
      echo "OOM or failure for $SIZE ctx=$C — skipping"
    fi

  done
done
echo "Profiling complete!"