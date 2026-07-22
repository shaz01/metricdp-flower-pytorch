#!/usr/bin/env bash
set -u
cd /teamspace/studios/this_studio
mkdir -p results/{runs,logs,evaluations,tmp}
cat > results/paper_reproduction_results.md <<'EOF'
# Non-CIA Paper Reproduction — H100 Detailed Evaluation

## Protocol

- 122 final global models: 90 homogeneous main, 18 non-IID main, 12 Appendix-B noise=0.003, and 2 Appendix-B FedAvg global-DP high-noise checks.
- Main settings: 4 clients, 20 rounds, 5 local epochs, batch 32, learning rate 0.001, clipping norm 5, noise 0.01.
- Homogeneous seeds: 42–46. Non-IID and Appendix-B seed: 42.
- Final model is saved transiently, evaluated on the server final-test split and all client held-out splits, then deleted after JSON/NPZ validation.
- CIA experiments are excluded.
- Previous accuracy-only results remain under `results-old-20260722-160446/`.
- H100 execution uses two concurrent experiment lanes; each experiment runs four concurrent clients.

## Artifact schema

- Run history: `results/runs/<run>.json`
- Detailed metrics: `results/evaluations/<run>.evaluation.json`
- Raw labels/probabilities/predictions: `results/evaluations/<run>.predictions.npz`
- Full log: `results/logs/<run>.log`

## Per-run results

| Run | Status | Final accuracy | Macro F1 | Macro precision | Macro OVR AUC | Duration | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
EOF
printf 'run\tstatus\texit_code\tduration_seconds\taccuracy\tloss\tmacro_f1\tmacro_precision\tmacro_auc\trun_json\tevaluation_json\tpredictions\tlog\tnote\n' > results/paper_reproduction_manifest.tsv
: > results/tasks.tsv
aggs=(fedavg fedavgm fedmedian fedprox fedopt fedyogi)
privacies=(vanilla global-dp metric-privacy)
for seed in 42 43 44 45 46; do for privacy in "${privacies[@]}"; do for agg in "${aggs[@]}"; do printf 'homogeneous\t%s\t%s\t%s\t0.01\tmain\n' "$privacy" "$agg" "$seed" >> results/tasks.tsv; done; done; done
for privacy in "${privacies[@]}"; do for agg in "${aggs[@]}"; do printf 'non-iid\t%s\t%s\t42\t0.01\tmain\n' "$privacy" "$agg" >> results/tasks.tsv; done; done
for privacy in global-dp metric-privacy; do for agg in "${aggs[@]}"; do printf 'homogeneous\t%s\t%s\t42\t0.003\tappendix-b\n' "$privacy" "$agg" >> results/tasks.tsv; done; done
printf 'homogeneous\tglobal-dp\tfedavg\t42\t0.05\tappendix-b\n' >> results/tasks.tsv
printf 'homogeneous\tglobal-dp\tfedavg\t42\t0.1\tappendix-b\n' >> results/tasks.tsv
[[ $(wc -l < results/tasks.tsv) -eq 122 ]] || { echo 'Task count is not 122' >&2; exit 2; }

echo "START_UTC=$(date -u +%FT%TZ)" | tee results/orchestrator_status.txt
# First pass and two recovery passes. Completed detailed artifacts are validated and skipped.
for pass in 1 2 3; do
  echo "PASS_${pass}_START=$(date -u +%FT%TZ)" | tee -a results/orchestrator_status.txt
  xargs -P 2 -n 6 ./results/run_one_detailed.sh < results/tasks.tsv 2>&1 | tee "results/logs/orchestrator-pass-${pass}.log"
  echo "PASS_${pass}_END=$(date -u +%FT%TZ)" | tee -a results/orchestrator_status.txt
done
uv run python results/summarize_detailed_results.py
rc=$?
echo "END_UTC=$(date -u +%FT%TZ) SUMMARY_RC=$rc" | tee -a results/orchestrator_status.txt
echo __H100_MATRIX_DONE__
exit $rc
