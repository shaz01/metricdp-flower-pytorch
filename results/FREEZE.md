# Frozen experiments

On 2026-10-02 every experiment that ran before the Aug 4 determinism fixes was removed from the
working tree. 

## Where the frozen state lives

The annotated tag **`freeze/2026-10-02`** points at `e6419e8`, the last master commit before the
reorganization. 

## Frozen paths

### Results (`results/…`)

- `8client_scaling/`, `48client_scaling/`: 8- vs 48-client scaling
- `noise_sweep/`: 8-client noise-multiplier sweep
- `sigma_calibration/`: 48-client sigma-calibrated FedYogi sweep
- `reproduce_paper/`, `reproduce_paper_failed/`, `reproduce/`: non-CIA paper reproduction
- `probe_fedopt_fedyogi/`, `fedyogi_rerun_4clients/`: FedOpt/FedYogi probes
- `cia_client_scaling/`: first-round CIA at 48 clients
- `noise_floor_check/`, `noise_floor_check_noniid/`: MPS noise-floor checks (scaling diagnosis)
- `no_agg_gradient_compare/`, `no_agg_raw_gradient_compare/`: no-aggregation gradient comparison
- `scale_controlled/`, `scale_controlled_epochs/`: constant-compute scaling (CUDA redo)
- `archive/`: constant-compute scaling, MPS-era v1/v2
- `noise_by_clients/`: noise-multiplier × client-count sweep
- `cia/check_determinism/`: Aug 4 Colab check that two same-seed runs match exactly (frozen later, on owner request)

### Reports (`reports/…`)

- `paper_reproduction.md`: Non-CIA Paper Reproduction (H100 detailed evaluation)
- `client_count_scaling.md`: Client-Count Scaling: 8 vs 48 Clients
- `constant_compute_scaling.md` / `.tex`: Constant-Compute Client-Count Scaling (CUDA redo)
- `archive/constant_compute_scaling_mps_v1v2.md`: the same, MPS-era
- `noise_by_clients.md`: Does the Noise-Multiplier Sweet Spot Shift with Client Count?
- `first_round_cia.md`: Client Inference Attack status report (first-round CIA, 48 clients)
- `no_aggregation_gradient_comparison.md`: No-Aggregation Client Model Divergence at 20 Clients
- `epoch_round_scaling.tex`: Epoch- and Round-Scaling Experiments
- `progress_report_phase1.tex` / `.pdf`: Phase-1 progress report (client-count scaling failure)

### Code (only produced frozen results)

- `experiments/client_scaling/sweep_8_clients.py`, `sweep_48_clients.py`: 8/48-client scaling
- `experiments/client_scaling/sweep_noise_multiplier.py`: noise sweep
- `experiments/client_scaling/sweep_48_sigma_calibrated.py`: sigma calibration
- `experiments/client_scaling/sweep_scale_controlled.py`, `sweep_scale_controlled_epochs.py`: constant-compute scaling
- `experiments/client_scaling/sweep_noise_by_clients.py`: noise-by-clients
- `experiments/client_scaling/scripts/contest_4_client.py`: 4-client contest run
- `experiments/cia/scripts/check_determinism.py`: determinism check
- `experiments/cia/scripts/contest.py` and its test in `experiments/cia/tests/test_shadow_dataset.py`: contest CIA matrix

`experiments/client_scaling/sweep_runner.py` is still used and was kept.
