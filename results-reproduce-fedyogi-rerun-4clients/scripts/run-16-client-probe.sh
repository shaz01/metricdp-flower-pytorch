#!/usr/bin/env bash
cd /teamspace/studios/this_studio
root="$PWD/results-reproduce-matrix-16clients-no-fedopt"; out="$root/runs"; logs="$root/logs"; mkdir -p "$out" "$logs"
name="clients-16__main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-42"
start=$(date +%s)
uv run python -m experiments.reproduce.runner --partition homogeneous --privacy metric-privacy --aggregation fedyogi --seed 42 --rounds 20 --local-epochs 5 --batch-size 32 --learning-rate 0.001 --noise-multiplier 0.01 --clipping-norm 5 --num-clients 16 --initialization-epochs 20 --initialization-batch-size 32 --initialization-learning-rate 0.001 --output-dir "$out" --run-name "$name" --max-parallel-clients 4 --client-cpus 2 --client-gpus 0.25 > "$logs/$name.log" 2>&1
rc=$?; duration=$(($(date +%s)-start)); echo "EXIT_CODE=$rc DURATION_SECONDS=$duration" >> "$logs/$name.log"; echo "PROBE_RC=$rc DURATION=$duration" | tee "$root/probe-status.txt"
exit $rc
