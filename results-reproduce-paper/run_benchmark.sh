#!/bin/bash
set -o pipefail
cd /teamspace/studios/this_studio
name=main__homogeneous__vanilla__fedavg__noise-0.01__seed-42
start=$(date +%s)
{
  echo "START_UTC=$(date -u +%FT%TZ)"
  uv run python -m experiments.reproduce.runner --partition homogeneous --privacy vanilla --aggregation fedavg --seed 42 --rounds 20 --local-epochs 5 --batch-size 32 --learning-rate 0.001 --noise-multiplier 0.01 --clipping-norm 5 --num-clients 4 --initialization-epochs 20 --initialization-batch-size 32 --initialization-learning-rate 0.001 --output-dir results-reproduce-step1/runs --run-name "$name" --max-parallel-clients 4 --client-cpus 2 --client-gpus 0.25 --save-model
  rc=$?
  if [ $rc -eq 0 ]; then
    uv run python results-reproduce-step1/evaluate_saved_model.py --model "results/runs/$name.pt" --run-json "results/runs/$name.json" --evaluation-json "results/evaluations/$name.evaluation.json" --predictions "results/evaluations/$name.predictions.npz"
    rc=$?
  fi
  if [ $rc -eq 0 ]; then rm "results/runs/$name.pt"; fi
  echo "END_UTC=$(date -u +%FT%TZ) EXIT_CODE=$rc DURATION_SECONDS=$(($(date +%s)-start))"
} > "results/logs/$name.log" 2>&1
echo "BENCHMARK_RC=$rc DURATION=$(($(date +%s)-start))" | tee results-reproduce-step1/benchmark_status.txt
exit $rc
