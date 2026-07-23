#!/usr/bin/env bash
set -u
cd /teamspace/studios/this_studio
out="$PWD/results-probe-fedopt-fedyogi/runs"; logs="$PWD/results-probe-fedopt-fedyogi/logs"; mkdir -p "$out" "$logs"
for agg in fedopt fedyogi; do
  name="probe__homogeneous__metric-privacy__${agg}__noise-0.01__seed-42"
  start=$(date +%s)
  {
    echo "START_UTC=$(date -u +%FT%TZ)"
    uv run python -m experiments.reproduce.runner --partition homogeneous --privacy metric-privacy --aggregation "$agg" --seed 42 --rounds 20 --local-epochs 5 --batch-size 32 --learning-rate 0.001 --noise-multiplier 0.01 --clipping-norm 5 --num-clients 4 --initialization-epochs 20 --initialization-batch-size 32 --initialization-learning-rate 0.001 --output-dir "$out" --run-name "$name" --max-parallel-clients 4 --client-cpus 2 --client-gpus 0.25
    rc=$?
    echo "END_UTC=$(date -u +%FT%TZ) EXIT_CODE=$rc DURATION_SECONDS=$(($(date +%s)-start))"
  } > "$logs/$name.log" 2>&1
  rc=$?
  echo "$name rc=$rc duration=$(($(date +%s)-start))s"
done
uv run python - <<'PY'
import glob,json,statistics
from pathlib import Path
out=[]
for p in sorted(glob.glob('results-probe-fedopt-fedyogi/runs/*.json')):
 d=json.load(open(p)); s=d['server_evaluate_metrics']; c=d['client_evaluate_metrics']; t=d['train_metrics']; rounds=sorted(int(k) for k in s if int(k)>0); last=rounds[-1]
 distances=[float(t[str(r)]['metric-dp-distance']) for r in rounds if 'metric-dp-distance' in t[str(r)]]
 row={'run_name':d['metadata']['run_name'],'completed_rounds':len(rounds),'final_server':s[str(last)],'final_client':c[str(last)],'last5_server_accuracy_mean':statistics.mean(float(s[str(r)]['accuracy']) for r in rounds[-5:]),'distance_min':min(distances) if distances else None,'distance_final':distances[-1] if distances else None,'all_distances_positive':all(x>0 for x in distances)}
 out.append(row)
Path('results-probe-fedopt-fedyogi/probe-summary.json').write_text(json.dumps(out,indent=2)+'\n')
md=['# FedOpt/FedYogi Metric-Privacy Probe','','Settings: homogeneous clients, seed 42, metric privacy, noise 0.01, C=5, 20 rounds, 5 local epochs.','', '| Aggregator | Rounds | Final server accuracy | Final server loss | Last-5 server accuracy | Min distance | Final distance | All distances > 0 |','|---|---:|---:|---:|---:|---:|---:|---|']
for r in out:
 a=r['run_name'].split('__')[3]; fs=r['final_server']; md.append(f"| {a} | {r['completed_rounds']} | {fs['accuracy']:.6f} | {fs['loss']:.6f} | {r['last5_server_accuracy_mean']:.6f} | {r['distance_min']:.8g} | {r['distance_final']:.8g} | {r['all_distances_positive']} |")
Path('results-probe-fedopt-fedyogi/README.md').write_text('\n'.join(md)+'\n')
print(json.dumps(out,indent=2))
PY
echo __PROBE_DONE__
