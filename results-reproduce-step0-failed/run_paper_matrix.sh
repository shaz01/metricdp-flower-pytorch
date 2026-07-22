#!/usr/bin/env bash
set -u
cd /teamspace/studios/this_studio
OUT="$PWD/results/paper_reproduction_runs"
LOG="$PWD/results/paper_reproduction_logs"
REPORT="$PWD/results/paper_reproduction_results.md"
MANIFEST="$PWD/results/paper_reproduction_manifest.tsv"
mkdir -p "$OUT" "$LOG"
printf 'run\tstatus\texit_code\tduration_seconds\taccuracy\tloss\tjson\tlog\n' > "$MANIFEST"

append_event() {
  printf '\n- `%s UTC` — %s\n' "$(date -u +'%Y-%m-%d %H:%M:%S')" "$1" >> "$REPORT"
}

record() {
  local name="$1" status="$2" rc="$3" duration="$4" json="$5" logfile="$6"
  uv run python - "$name" "$status" "$rc" "$duration" "$json" "$logfile" "$MANIFEST" "$REPORT" <<'PY'
import json, sys
from pathlib import Path
name,status,rc,duration,jpath,lpath,manifest,report=sys.argv[1:]
acc=loss=""
note=""
if status == "PASS":
    try:
        d=json.loads(Path(jpath).read_text())
        hist=d["server_evaluate_metrics"]
        rounds=[int(k) for k in hist if int(k)>0]
        final=hist[str(max(rounds))] if rounds else hist.get("0",{})
        acc=final.get("accuracy","")
        loss=final.get("loss","")
    except Exception as e:
        status="FAIL"
        note=f"result parse failed: {e}"
with Path(manifest).open("a") as f:
    f.write("\t".join(map(str,[name,status,rc,duration,acc,loss,jpath,lpath]))+"\n")
with Path(report).open("a") as f:
    f.write(f"| `{name}` | {status} | {acc} | {loss} | {note or f'{duration}s; exit {rc}'} |\n")
PY
}

run_one() {
  local partition="$1" privacy="$2" agg="$3" seed="$4" noise="$5" family="$6"
  local name="${family}__${partition}__${privacy}__${agg}__noise-${noise}__seed-${seed}"
  local json="$OUT/$name.json" log="$LOG/$name.log"
  local start end rc status
  if [[ -s "$json" && -s "$log" ]] && grep -q '^EXIT_CODE=0$' "$log"; then
    record "$name" PASS 0 current-attempt "$json" "$log"
    append_event "VALIDATED current-attempt result $name — PASS (not rerun)"
    echo "[$(date -u +'%FT%TZ')] PASS $name validated-current-attempt"
    return 0
  fi
  append_event "STARTED $name"
  start=$(date +%s)
  local cmd=(uv run python -m experiments.reproduce.runner
    --partition "$partition" --privacy "$privacy" --aggregation "$agg"
    --seed "$seed" --rounds 20 --local-epochs 5 --batch-size 32
    --learning-rate 0.001 --noise-multiplier "$noise" --clipping-norm 5
    --num-clients 4 --initialization-epochs 20 --initialization-batch-size 32
    --initialization-learning-rate 0.001 --output-dir "$OUT" --run-name "$name"
    --max-parallel-clients 4 --client-cpus 2 --client-gpus 0.25)
  (
    printf 'START_UTC=%s\n' "$(date -u +'%FT%TZ')"
    printf 'COMMAND='; printf '%q ' "${cmd[@]}"; printf '\n'
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
    "${cmd[@]}"
    rc=$?
    printf 'EXIT_CODE=%s\nEND_UTC=%s\n' "$rc" "$(date -u +'%FT%TZ')"
    exit "$rc"
  ) >"$log" 2>&1
  rc=$?
  end=$(date +%s)
  if [[ $rc -eq 0 && -s "$json" ]]; then status=PASS; else status=FAIL; fi
  record "$name" "$status" "$rc" "$((end-start))" "$json" "$log"
  append_event "COMPLETED $name — $status (exit $rc, $((end-start))s)"
  echo "[$(date -u +'%FT%TZ')] $status $name rc=$rc duration=$((end-start))s"
  return 0
}

append_event "Unit tests PASS: 26 passed, 5 deselected. CUDA smoke PASS: final centralized accuracy 0.5, loss 1.65556001663208. Matrix launch uses four concurrent Ray clients at 0.25 L4 GPU each."

aggs=(fedavg fedavgm fedmedian fedprox fedopt fedyogi)
privacies=(vanilla global-dp metric-privacy)

# Main homogeneous matrix: five explicit seeds.
for seed in 42 43 44 45 46; do
  for privacy in "${privacies[@]}"; do
    for agg in "${aggs[@]}"; do
      run_one homogeneous "$privacy" "$agg" "$seed" 0.01 main
    done
  done
done

# Main non-IID fixed-seed matrix.
for privacy in "${privacies[@]}"; do
  for agg in "${aggs[@]}"; do
    run_one non-iid "$privacy" "$agg" 42 0.01 main
  done
done

# Appendix B, noise 0.003 for both private mechanisms.
for privacy in global-dp metric-privacy; do
  for agg in "${aggs[@]}"; do
    run_one homogeneous "$privacy" "$agg" 42 0.003 appendix-b
  done
done

# Appendix B FedAvg global-DP non-convergence checks.
run_one homogeneous global-dp fedavg 42 0.05 appendix-b
run_one homogeneous global-dp fedavg 42 0.1 appendix-b

uv run python - "$MANIFEST" "$REPORT" <<'PY'
import csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path
manifest, report=map(Path,sys.argv[1:])
rows=list(csv.DictReader(manifest.open(),delimiter='\t'))
passed=[r for r in rows if r['status']=='PASS']
failed=[r for r in rows if r['status']!='PASS']
with report.open('a') as f:
    f.write("\n## Final execution summary\n\n")
    f.write(f"- Requested matrix runs: {len(rows)}\n- Passed: {len(passed)}\n- Failed: {len(failed)}\n")
    f.write("- F1 and precision: unavailable because the implementation does not persist predictions/confusion statistics; no values were inferred.\n")
    if failed:
        f.write("\n### Failures\n")
        for r in failed: f.write(f"- `{r['run']}`: exit {r['exit_code']}; log `{r['log']}`\n")
    groups=defaultdict(list)
    for r in passed:
        if r['run'].startswith('main__homogeneous') and r['accuracy']:
            parts=r['run'].split('__')
            groups[(parts[2],parts[3])].append(float(r['accuracy']))
    f.write("\n### Homogeneous five-seed final centralized accuracy\n\n| Privacy | Aggregator | Mean | Std | n |\n|---|---|---:|---:|---:|\n")
    for (privacy,agg), vals in sorted(groups.items()):
        std=statistics.stdev(vals) if len(vals)>1 else 0.0
        f.write(f"| {privacy} | {agg} | {statistics.mean(vals):.6f} | {std:.6f} | {len(vals)} |\n")
PY
append_event "ALL REQUESTED NON-CIA RUNS FINISHED"
echo __MATRIX_DONE__
