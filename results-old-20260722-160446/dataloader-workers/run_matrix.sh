#!/usr/bin/env bash
set -uo pipefail

ROOT=/teamspace/studios/this_studio
BASE="$ROOT/results/dataloader-workers"
OUT="$BASE/runs"
LOGS="$BASE/logs"
REPORT="$BASE/results.md"
mkdir -p "$OUT" "$LOGS"
cd "$ROOT"

append() { printf '%s\n' "$*" >> "$REPORT"; }

append ""
append "## Full matrix"
append ""
append "- Started: $(date -u +%FT%TZ)"
append "- Main homogeneous: five seeds (42–46), 3 privacy modes × 6 aggregators."
append "- Main non-IID: fixed seed 42, 3 privacy modes × 6 aggregators."
append "- Appendix B: homogeneous GDP/metric privacy × 6 aggregators at noise 0.003; GDP FedAvg at 0.05 and 0.1."
append "- Common settings: 4 clients, 20 rounds, 5 local epochs, batch 32, Adam LR 0.001, clipping norm 5, stateful initialization 20 epochs."
append "- Runtime: two concurrent clients, 0.5 L4 GPU per client."
append "- CIA excluded. The implementation records accuracy/loss, not F1/precision predictions."
append ""
append "| Run | Status | Seconds | Server accuracy | Server loss | Client accuracy last 5 (mean ± std) |"
append "|---|---:|---:|---:|---:|---:|"

summarize() {
  local json="$1"
  uv run python - "$json" <<'PY'
import json, math, statistics, sys
p=sys.argv[1]
d=json.load(open(p))
server=d.get("server_evaluate_metrics", {})
rounds=sorted((int(k),v) for k,v in server.items() if int(k)>0)
final=rounds[-1][1] if rounds else {}
client=d.get("client_evaluate_metrics", {})
cr=sorted((int(k),v) for k,v in client.items() if int(k)>0)[-5:]
vals=[]
for _,v in cr:
    x=v.get("eval_acc", v.get("accuracy"))
    if x is not None: vals.append(float(x))
mean=statistics.mean(vals) if vals else math.nan
std=statistics.pstdev(vals) if len(vals)>1 else (0.0 if vals else math.nan)
print(f'{float(final.get("accuracy", math.nan)):.6f}\t{float(final.get("loss", math.nan)):.6f}\t{mean:.6f} ± {std:.6f}')
PY
}

run_one() {
  local partition="$1" privacy="$2" agg="$3" seed="$4" noise="$5" tag="$6"
  local name="${tag}__${partition}__${privacy}__${agg}__seed-${seed}__noise-${noise}"
  local json="$OUT/$name.json" log="$LOGS/$name.log"
  if [[ -s "$json" ]]; then
    local summary
    summary=$(summarize "$json" 2>/dev/null || printf 'nan\tnan\tnan')
    IFS=$'\t' read -r acc loss client <<< "$summary"
    append "| $name | existing | 0 | $acc | $loss | $client |"
    return 0
  fi
  local start end rc summary acc loss client
  start=$(date +%s)
  echo "[$(date -u +%FT%TZ)] START $name" | tee "$log"
  set +e
  uv run python -m experiments.reproduce.runner \
    --partition "$partition" \
    --privacy "$privacy" \
    --aggregation "$agg" \
    --seed "$seed" \
    --noise-multiplier "$noise" \
    --clipping-norm 5 \
    --num-clients 4 \
    --rounds 20 \
    --local-epochs 5 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --initialization-epochs 20 \
    --initialization-batch-size 32 \
    --initialization-learning-rate 0.001 \
    --max-parallel-clients 2 \
    --client-cpus 1 \
    --client-gpus 0.5 \
    --output-dir "$OUT" \
    --run-name "$name" >> "$log" 2>&1
  rc=$?
  set -e
  end=$(date +%s)
  if [[ $rc -eq 0 && -s "$json" ]]; then
    summary=$(summarize "$json" 2>/dev/null || printf 'nan\tnan\tnan')
    IFS=$'\t' read -r acc loss client <<< "$summary"
    append "| $name | ok | $((end-start)) | $acc | $loss | $client |"
  else
    append "| $name | FAILED($rc) | $((end-start)) | — | — | — |"
  fi
  echo "[$(date -u +%FT%TZ)] END $name rc=$rc elapsed=$((end-start))s"
  ray stop --force >/dev/null 2>&1 || true
  return 0
}

trap 'append ""; append "- Interrupted: $(date -u +%FT%TZ)"' INT TERM

aggregations=(fedavg fedavgm fedmedian fedprox fedopt fedyogi)
privacies=(vanilla global-dp metric-privacy)

for seed in 42 43 44 45 46; do
  for privacy in "${privacies[@]}"; do
    for agg in "${aggregations[@]}"; do
      run_one homogeneous "$privacy" "$agg" "$seed" 0.01 main
    done
  done
done

for privacy in "${privacies[@]}"; do
  for agg in "${aggregations[@]}"; do
    run_one non-iid "$privacy" "$agg" 42 0.01 main
  done
done

for privacy in global-dp metric-privacy; do
  for agg in "${aggregations[@]}"; do
    run_one homogeneous "$privacy" "$agg" 42 0.003 appendix-b
  done
done

run_one homogeneous global-dp fedavg 42 0.05 appendix-b
run_one homogeneous global-dp fedavg 42 0.1 appendix-b

append ""
append "- Finished: $(date -u +%FT%TZ)"
append "- JSON results: \`results/dataloader-workers/runs/\`"
append "- Per-run logs: \`results/dataloader-workers/logs/\`"
echo __MATRIX_COMPLETE__
