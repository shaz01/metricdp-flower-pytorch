#!/usr/bin/env bash
set -u
cd /teamspace/studios/this_studio
while [[ -e results-reproduce-new-matrices.PAUSE ]]; do sleep 5; done
root="$1"; clients="$2"; partition="$3"; privacy="$4"; agg="$5"; seed="$6"; noise="$7"; family="$8"
out="$PWD/$root/runs"; logs="$PWD/$root/logs"; mkdir -p "$out" "$logs"
name="clients-${clients}__${family}__${partition}__${privacy}__${agg}__noise-${noise}__seed-${seed}"
run="$out/$name.json"; evaluation="$out/$name.evaluation.json"; predictions="$out/$name.predictions.npz"; log="$logs/$name.log"
valid() { uv run python - "$run" "$evaluation" "$predictions" "$name" "$clients" <<'PY' >/dev/null 2>&1
import json,sys,numpy as np
from pathlib import Path
rp,ep,pp,name,clients=sys.argv[1:]; r=json.loads(Path(rp).read_text()); e=json.loads(Path(ep).read_text()); h=r['server_evaluate_metrics']; fr=max(int(k) for k in h if int(k)>0)
assert r['metadata']['run_name']==name and int(r['metadata']['num_clients'])==int(clients) and e['run_name']==name
assert len(e['clients'])==int(clients) and abs(float(h[str(fr)]['accuracy'])-float(e['server_final_test']['accuracy']))<1e-12
with np.load(pp) as p: assert p['server_probabilities'].shape[1]==4
PY
}
record() {
 local status="$1" rc="$2" seconds="$3" note="$4" line
 line=$(uv run python - "$name" "$root" "$status" "$rc" "$seconds" "$run" "$evaluation" "$predictions" "$log" "$note" <<'PY'
import json,sys
from pathlib import Path
name,root,status,rc,seconds,rp,ep,pp,lp,note=sys.argv[1:]; acc=f1=precision=auc=''
if status=='PASS':
 e=json.loads(Path(ep).read_text()); s=e['server_final_test']; acc=s['accuracy']; f1=s['averages']['macro']['f1']; precision=s['averages']['macro']['precision']; auc=s['roc_ovr']['macro_auc']
print('\t'.join(map(str,[name,root,status,rc,seconds,acc,f1,precision,auc,rp,ep,pp,lp,note])))
PY
)
 { flock 9; if ! awk -F '\t' -v n="$name" '$1==n&&$3=="PASS"{x=1}END{exit !x}' results-reproduce-new-matrices-manifest.tsv 2>/dev/null; then printf '%s\n' "$line" >> results-reproduce-new-matrices-manifest.tsv; fi; printf '| `%s` | %s | %s | %s | %s | %s | %ss | %s |\n' "$name" "$status" "$(echo "$line"|cut -f6)" "$(echo "$line"|cut -f7)" "$(echo "$line"|cut -f8)" "$(echo "$line"|cut -f9)" "$seconds" "$note" >> results-reproduce-new-matrices.md; } 9>results-reproduce-new-matrices.lock
}
if valid; then record PASS 0 0 validated-existing; echo "PASS $name existing"; exit 0; fi
rm -f "$run" "$evaluation" "$predictions"
start=$(date +%s)
uv run python -m experiments.reproduce.runner --partition "$partition" --privacy "$privacy" --aggregation "$agg" --seed "$seed" --rounds 20 --local-epochs 5 --batch-size 32 --learning-rate 0.001 --noise-multiplier "$noise" --clipping-norm 5 --num-clients "$clients" --initialization-epochs 20 --initialization-batch-size 32 --initialization-learning-rate 0.001 --output-dir "$out" --run-name "$name" --max-parallel-clients 16 --client-cpus 1 --client-gpus 0.0625 > "$log" 2>&1
rc=$?; seconds=$(($(date +%s)-start)); echo "EXIT_CODE=$rc DURATION_SECONDS=$seconds" >> "$log"
if [[ $rc -eq 0 ]] && valid; then record PASS 0 "$seconds" trained-evaluated; echo "PASS $name ${seconds}s"; else record FAIL "$rc" "$seconds" failed-see-log; echo "FAIL $name rc=$rc ${seconds}s"; fi
exit 0
