#!/usr/bin/env bash
set -u
cd /teamspace/studios/this_studio
root4=results-reproduce-fedyogi-rerun-4clients
root16=results-reproduce-matrix-16clients-no-fedopt
mkdir -p "$root4"/{runs,logs} "$root16"/{runs,logs}
cat > results-reproduce-new-matrices.md <<'EOF'
# FedYogi 4-client rerun and 16-client no-FedOpt matrix

## Scope

- 20 four-client FedYogi reruns from the paper's 120-run matrix.
- 100 sixteen-client runs using FedAvg, FedAvgM, FedMedian, FedProx, and FedYogi; FedOpt excluded.
- CIA excluded.
- Settings retained: 20 rounds, 5 local epochs, batch 32, LR 0.001, C=5, main noise 0.01, Appendix-B noise 0.003, initialization 20 epochs.
- Sixteen-client `auto` partitions are scalable rather than the exact four-client paper count matrix.
- Optimized H100 scheduling: one experiment lane; sixteen clients concurrently; DataLoader workers disabled to avoid short-lived worker overhead.

## Runs

| Run | Status | Accuracy | Macro F1 | Macro precision | Macro AUC | Duration | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
EOF
printf 'run\troot\tstatus\texit_code\tduration_seconds\taccuracy\tmacro_f1\tmacro_precision\tmacro_auc\trun_json\tevaluation_json\tpredictions\tlog\tnote\n' > results-reproduce-new-matrices-manifest.tsv
: > results-reproduce-new-matrices-tasks.tsv
privacies=(vanilla global-dp metric-privacy); aggs16=(fedavg fedavgm fedmedian fedprox fedyogi)
# Four-client FedYogi: 15 homogeneous + 3 non-IID + 2 Appendix B = 20.
for seed in 42 43 44 45 46; do for privacy in "${privacies[@]}"; do printf '%s\t4\thomogeneous\t%s\tfedyogi\t%s\t0.01\tmain\n' "$root4" "$privacy" "$seed" >> results-reproduce-new-matrices-tasks.tsv; done; done
for privacy in "${privacies[@]}"; do printf '%s\t4\tnon-iid\t%s\tfedyogi\t42\t0.01\tmain\n' "$root4" "$privacy" >> results-reproduce-new-matrices-tasks.tsv; done
for privacy in global-dp metric-privacy; do printf '%s\t4\thomogeneous\t%s\tfedyogi\t42\t0.003\tappendix-b\n' "$root4" "$privacy" >> results-reproduce-new-matrices-tasks.tsv; done
# Sixteen-client matrix without FedOpt: 75 homogeneous + 15 non-IID + 10 Appendix B = 100.
for seed in 42 43 44 45 46; do for privacy in "${privacies[@]}"; do for agg in "${aggs16[@]}"; do printf '%s\t16\thomogeneous\t%s\t%s\t%s\t0.01\tmain\n' "$root16" "$privacy" "$agg" "$seed" >> results-reproduce-new-matrices-tasks.tsv; done; done; done
for privacy in "${privacies[@]}"; do for agg in "${aggs16[@]}"; do printf '%s\t16\tnon-iid\t%s\t%s\t42\t0.01\tmain\n' "$root16" "$privacy" "$agg" >> results-reproduce-new-matrices-tasks.tsv; done; done
for privacy in global-dp metric-privacy; do for agg in "${aggs16[@]}"; do printf '%s\t16\thomogeneous\t%s\t%s\t42\t0.003\tappendix-b\n' "$root16" "$privacy" "$agg" >> results-reproduce-new-matrices-tasks.tsv; done; done
[[ $(wc -l < results-reproduce-new-matrices-tasks.tsv) -eq 120 ]] || exit 2
echo "START_UTC=$(date -u +%FT%TZ)" > results-reproduce-new-matrices-status.txt
for pass in 1 2 3; do
 echo "PASS_${pass}_START=$(date -u +%FT%TZ)" | tee -a results-reproduce-new-matrices-status.txt
 xargs -P 1 -n 8 ./run_one_clients_matrix.sh < results-reproduce-new-matrices-tasks.tsv 2>&1 | tee "results-reproduce-new-matrices-pass-${pass}.log"
 echo "PASS_${pass}_END=$(date -u +%FT%TZ)" | tee -a results-reproduce-new-matrices-status.txt
done
uv run python summarize_new_matrices.py
rc=$?
echo "END_UTC=$(date -u +%FT%TZ) SUMMARY_RC=$rc" | tee -a results-reproduce-new-matrices-status.txt
echo __NEW_MATRICES_DONE__
exit $rc
