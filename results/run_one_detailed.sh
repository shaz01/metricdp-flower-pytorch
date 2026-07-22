#!/usr/bin/env bash
set -u
cd /teamspace/studios/this_studio
partition="$1"; privacy="$2"; agg="$3"; seed="$4"; noise="$5"; family="$6"
name="${family}__${partition}__${privacy}__${agg}__noise-${noise}__seed-${seed}"
rundir="$PWD/results/runs"; evaldir="$PWD/results/evaluations"; logdir="$PWD/results/logs"
json="$rundir/$name.json"; model="$rundir/$name.pt"; evaluation="$evaldir/$name.evaluation.json"; predictions="$evaldir/$name.predictions.npz"; log="$logdir/$name.log"
mkdir -p "$rundir" "$evaldir" "$logdir" results/tmp

valid_artifacts() {
  uv run python - "$json" "$evaluation" "$predictions" <<'PY' >/dev/null 2>&1
import json,sys,numpy as np
from pathlib import Path
run,evaluation,predictions=map(Path,sys.argv[1:])
r=json.loads(run.read_text()); e=json.loads(evaluation.read_text())
assert e['run_name']==r['metadata']['run_name']
h=r['server_evaluate_metrics']; fr=max(int(k) for k in h if int(k)>0)
assert abs(e['server_final_test']['accuracy']-float(h[str(fr)]['accuracy'])) < 1e-12
with np.load(predictions) as p:
 assert len(p['server_y_true'])==e['server_final_test']['num_examples']
 assert p['server_probabilities'].shape[1]==4
PY
}

record() {
  local status="$1" rc="$2" duration="$3" note="$4"
  local line
  line=$(uv run python - "$name" "$status" "$rc" "$duration" "$json" "$evaluation" "$predictions" "$log" "$note" <<'PY'
import json,sys
from pathlib import Path
name,status,rc,duration,jp,ep,pp,lp,note=sys.argv[1:]
acc=loss=f1=precision=auc=''
if status=='PASS':
 r=json.loads(Path(jp).read_text()); e=json.loads(Path(ep).read_text()); h=r['server_evaluate_metrics']; fr=max(int(k) for k in h if int(k)>0); m=h[str(fr)]
 acc=m.get('accuracy',''); loss=m.get('loss',''); s=e['server_final_test']; f1=s['averages']['macro']['f1']; precision=s['averages']['macro']['precision']; auc=s['roc_ovr']['macro_auc']
print('\t'.join(map(str,[name,status,rc,duration,acc,loss,f1,precision,auc,jp,ep,pp,lp,note])))
PY
)
  { flock 9; printf '%s\n' "$line" >> results/paper_reproduction_manifest.tsv; printf '| `%s` | %s | %s | %s | %s | %s | %ss | %s |\n' "$name" "$status" "$(echo "$line"|cut -f5)" "$(echo "$line"|cut -f7)" "$(echo "$line"|cut -f8)" "$(echo "$line"|cut -f9)" "$duration" "$note" >> results/paper_reproduction_results.md; } 9>results/report.lock
}

if valid_artifacts; then
  if ! awk -F '\t' -v n="$name" '$1==n && $2=="PASS" {found=1} END{exit !found}' results/paper_reproduction_manifest.tsv 2>/dev/null; then
    record PASS 0 0 validated-existing-detailed-artifacts
  fi
  echo "PASS $name existing"
  exit 0
fi

# If training completed previously and left its transient model, finish evaluation without retraining.
start=$(date +%s)
if [[ -s "$json" && -s "$model" ]]; then
  echo "[$(date -u +%FT%TZ)] RESUME_POSTPROCESS $name" >> "$log"
else
  rm -f "$json" "$model" "$evaluation" "$predictions"
  # Ray's Unix-domain socket path is capped at 107 bytes, so keep this short.
  tmpbase="/tmp/mdp-${BASHPID}"
  mkdir -p "$tmpbase"
  {
    echo "START_UTC=$(date -u +%FT%TZ)"
    echo "RUN_NAME=$name"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
    TMPDIR="$tmpbase" RAY_TMPDIR="$tmpbase/ray" uv run python -m experiments.reproduce.runner \
      --partition "$partition" --privacy "$privacy" --aggregation "$agg" --seed "$seed" \
      --rounds 20 --local-epochs 5 --batch-size 32 --learning-rate 0.001 \
      --noise-multiplier "$noise" --clipping-norm 5 --num-clients 4 \
      --initialization-epochs 20 --initialization-batch-size 32 --initialization-learning-rate 0.001 \
      --output-dir "$rundir" --run-name "$name" --max-parallel-clients 4 \
      --client-cpus 2 --client-gpus 0.25 --save-model
  } > "$log" 2>&1
  train_rc=$?
  if [[ $train_rc -ne 0 ]]; then
    duration=$(($(date +%s)-start)); record FAIL "$train_rc" "$duration" training-failed; echo "FAIL $name training rc=$train_rc"; exit 0
  fi
fi

uv run python results/evaluate_saved_model.py --model "$model" --run-json "$json" --evaluation-json "$evaluation" --predictions "$predictions" >> "$log" 2>&1
eval_rc=$?
duration=$(($(date +%s)-start))
if [[ $eval_rc -eq 0 ]] && valid_artifacts; then
  rm -f "$model"
  echo "END_UTC=$(date -u +%FT%TZ) EXIT_CODE=0 DURATION_SECONDS=$duration" >> "$log"
  record PASS 0 "$duration" trained-evaluated-model-deleted
  echo "PASS $name ${duration}s"
else
  echo "END_UTC=$(date -u +%FT%TZ) EVALUATION_EXIT_CODE=$eval_rc DURATION_SECONDS=$duration" >> "$log"
  record FAIL "$eval_rc" "$duration" evaluation-failed-model-retained
  echo "FAIL $name evaluation rc=$eval_rc"
fi
exit 0
