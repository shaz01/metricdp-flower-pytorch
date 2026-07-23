#!/usr/bin/env bash
set -u
cd /teamspace/studios/this_studio
out="$PWD/results-probe-fedopt-fedyogi/runs"; logs="$PWD/results-probe-fedopt-fedyogi/logs"; mkdir -p "$out" "$logs"
for privacy in vanilla global-dp; do
 name="probe__homogeneous__${privacy}__fedopt__noise-0.01__seed-42"; start=$(date +%s)
 uv run python -m experiments.reproduce.runner --partition homogeneous --privacy "$privacy" --aggregation fedopt --seed 42 --rounds 20 --local-epochs 5 --batch-size 32 --learning-rate 0.001 --noise-multiplier 0.01 --clipping-norm 5 --num-clients 4 --initialization-epochs 20 --initialization-batch-size 32 --initialization-learning-rate 0.001 --output-dir "$out" --run-name "$name" --max-parallel-clients 4 --client-cpus 2 --client-gpus 0.25 > "$logs/$name.log" 2>&1
 rc=$?; echo "END_UTC=$(date -u +%FT%TZ) EXIT_CODE=$rc DURATION_SECONDS=$(($(date +%s)-start))" >> "$logs/$name.log"; echo "$name rc=$rc duration=$(($(date +%s)-start))s"
done
echo __CONTROLS_DONE__
