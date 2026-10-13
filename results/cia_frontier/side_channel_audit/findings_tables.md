### Gate G1

| Part | Statistic | Value [95% interval] | Decision |
|---|---|---|---|
| G1(a) | blind noise-level AUC, metric privacy, top-decile targets, pooled over 6 cells | 0.838 [0.779, 0.902] | pass |
| (attribution) | oracle noise-level AUC, metric privacy, same targets | 0.839 [0.781, 0.903] | |
| (attribution) | blind noise-level AUC, global DP, same targets (update-norm channel) | 0.763 [0.706, 0.824] | |
| G1(b) | increment minus loss, TPR at 5% FPR, all targets, clean view, mean over 12 cells | +0.074 [-0.060, +0.116] | inconclusive |

Attribution rule (pre-run revision): metric-privacy noise level.

### Attacks per cell: clean view, top-decile targets

AUC; TPR at 5% FPR; TPR at 1% FPR; value [95% cluster-bootstrap interval].

| Dataset | Clients | Mechanism | Attacker | AUC | TPR@5% | TPR@1% |
|---|---|---|---|---|---|---|
| eurosat | 3 | global-dp | noise_level | 0.982 [0.967, 1.000] | 0.953 [0.907, 1.000] | 0.907 [0.827, 1.000] |
| eurosat | 3 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 3 | global-dp | increment | 0.784 [0.736, 1.000] | 0.280 [0.280, 1.000] | 0.280 [0.020, 1.000] |
| eurosat | 3 | global-dp | loss | 0.934 [0.877, 0.998] | 0.633 [0.333, 1.000] | 0.500 [0.207, 0.960] |
| eurosat | 3 | metric-privacy | noise_level | 0.996 [0.983, 1.000] | 0.967 [0.920, 1.000] | 0.960 [0.900, 1.000] |
| eurosat | 3 | metric-privacy | oracle_noise_level | 0.989 [0.974, 1.000] | 0.907 [0.853, 1.000] | 0.833 [0.707, 1.000] |
| eurosat | 3 | metric-privacy | increment | 0.882 [0.763, 1.000] | 0.647 [0.320, 1.000] | 0.647 [0.060, 1.000] |
| eurosat | 3 | metric-privacy | loss | 0.945 [0.901, 0.998] | 0.660 [0.367, 1.000] | 0.647 [0.353, 0.980] |
| eurosat | 5 | global-dp | noise_level | 0.918 [0.842, 0.964] | 0.767 [0.340, 0.940] | 0.160 [0.027, 0.920] |
| eurosat | 5 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 5 | global-dp | increment | 0.778 [0.778, 1.000] | 0.667 [0.333, 1.000] | 0.667 [0.333, 1.000] |
| eurosat | 5 | global-dp | loss | 0.784 [0.773, 0.980] | 0.513 [0.340, 0.980] | 0.493 [0.020, 0.980] |
| eurosat | 5 | metric-privacy | noise_level | 0.994 [0.991, 1.000] | 0.973 [0.947, 1.000] | 0.973 [0.947, 1.000] |
| eurosat | 5 | metric-privacy | oracle_noise_level | 0.997 [0.995, 1.000] | 0.987 [0.973, 1.000] | 0.967 [0.933, 1.000] |
| eurosat | 5 | metric-privacy | increment | 0.778 [0.778, 1.000] | 0.667 [0.333, 1.000] | 0.667 [0.333, 1.000] |
| eurosat | 5 | metric-privacy | loss | 0.783 [0.774, 0.981] | 0.493 [0.340, 0.980] | 0.467 [0.020, 0.980] |
| eurosat | 8 | global-dp | noise_level | 0.690 [0.690, 0.849] | 0.027 [0.013, 0.620] | 0.000 [0.000, 0.540] |
| eurosat | 8 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 8 | global-dp | increment | 0.574 [0.178, 1.000] | 0.000 [0.000, 1.000] | 0.000 [0.000, 1.000] |
| eurosat | 8 | global-dp | loss | 0.739 [0.739, 0.968] | 0.340 [0.313, 0.880] | 0.240 [0.240, 0.860] |
| eurosat | 8 | metric-privacy | noise_level | 0.891 [0.817, 1.000] | 0.687 [0.373, 1.000] | 0.520 [0.227, 1.000] |
| eurosat | 8 | metric-privacy | oracle_noise_level | 0.887 [0.829, 1.000] | 0.687 [0.373, 1.000] | 0.553 [0.233, 1.000] |
| eurosat | 8 | metric-privacy | increment | 0.676 [0.672, 1.000] | 0.207 [0.207, 1.000] | 0.207 [0.207, 1.000] |
| eurosat | 8 | metric-privacy | loss | 0.744 [0.744, 0.976] | 0.327 [0.313, 0.880] | 0.287 [0.287, 0.860] |
| alzheimer | 3 | global-dp | noise_level | 0.983 [0.969, 0.989] | 0.907 [0.820, 0.967] | 0.773 [0.700, 0.960] |
| alzheimer | 3 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 3 | global-dp | increment | 0.802 [0.654, 0.890] | 0.533 [0.320, 0.720] | 0.507 [0.320, 0.700] |
| alzheimer | 3 | global-dp | loss | 0.994 [0.989, 1.000] | 0.920 [0.907, 1.000] | 0.867 [0.780, 1.000] |
| alzheimer | 3 | metric-privacy | noise_level | 0.791 [0.742, 0.970] | 0.413 [0.240, 0.860] | 0.260 [0.060, 0.520] |
| alzheimer | 3 | metric-privacy | oracle_noise_level | 0.787 [0.738, 0.969] | 0.407 [0.220, 0.840] | 0.253 [0.060, 0.520] |
| alzheimer | 3 | metric-privacy | increment | 0.617 [0.593, 0.728] | 0.160 [0.027, 0.460] | 0.060 [0.013, 0.460] |
| alzheimer | 3 | metric-privacy | loss | 0.994 [0.991, 0.996] | 0.947 [0.913, 0.980] | 0.893 [0.860, 0.920] |
| alzheimer | 5 | global-dp | noise_level | 0.706 [0.698, 0.722] | 0.200 [0.120, 0.220] | 0.060 [0.020, 0.087] |
| alzheimer | 5 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 5 | global-dp | increment | 0.786 [0.783, 0.929] | 0.433 [0.240, 0.560] | 0.313 [0.040, 0.560] |
| alzheimer | 5 | global-dp | loss | 0.802 [0.778, 0.968] | 0.333 [0.267, 0.820] | 0.180 [0.140, 0.620] |
| alzheimer | 5 | metric-privacy | noise_level | 0.850 [0.824, 0.990] | 0.580 [0.313, 0.980] | 0.547 [0.307, 0.900] |
| alzheimer | 5 | metric-privacy | oracle_noise_level | 0.850 [0.824, 0.990] | 0.580 [0.313, 0.980] | 0.553 [0.307, 0.900] |
| alzheimer | 5 | metric-privacy | increment | 0.749 [0.597, 0.984] | 0.353 [0.133, 0.960] | 0.300 [0.020, 0.740] |
| alzheimer | 5 | metric-privacy | loss | 0.792 [0.754, 0.971] | 0.287 [0.287, 0.900] | 0.120 [0.013, 0.760] |
| alzheimer | 8 | global-dp | noise_level | 0.531 [0.511, 0.570] | 0.060 [0.020, 0.067] | 0.007 [0.000, 0.020] |
| alzheimer | 8 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 8 | global-dp | increment | 0.539 [0.115, 0.875] | 0.113 [0.000, 0.460] | 0.113 [0.000, 0.440] |
| alzheimer | 8 | global-dp | loss | 0.643 [0.603, 0.786] | 0.120 [0.120, 0.267] | 0.060 [0.020, 0.180] |
| alzheimer | 8 | metric-privacy | noise_level | 0.795 [0.671, 0.990] | 0.360 [0.200, 1.000] | 0.267 [0.060, 0.720] |
| alzheimer | 8 | metric-privacy | oracle_noise_level | 0.795 [0.670, 0.990] | 0.360 [0.200, 1.000] | 0.267 [0.060, 0.720] |
| alzheimer | 8 | metric-privacy | increment | 0.660 [0.530, 0.964] | 0.273 [0.187, 0.420] | 0.167 [0.080, 0.360] |
| alzheimer | 8 | metric-privacy | loss | 0.665 [0.612, 0.926] | 0.273 [0.080, 0.680] | 0.227 [0.020, 0.660] |

### Attacks per cell: clean view, all targets

AUC; TPR at 5% FPR; TPR at 1% FPR; value [95% cluster-bootstrap interval].

| Dataset | Clients | Mechanism | Attacker | AUC | TPR@5% | TPR@1% |
|---|---|---|---|---|---|---|
| eurosat | 3 | global-dp | noise_level | 0.991 [0.981, 0.999] | 0.976 [0.949, 0.998] | 0.924 [0.889, 0.982] |
| eurosat | 3 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 3 | global-dp | increment | 0.876 [0.783, 0.977] | 0.562 [0.307, 0.882] | 0.487 [0.269, 0.876] |
| eurosat | 3 | global-dp | loss | 0.799 [0.668, 0.988] | 0.069 [0.060, 0.945] | 0.031 [0.031, 0.802] |
| eurosat | 3 | metric-privacy | noise_level | 0.951 [0.912, 0.988] | 0.820 [0.571, 0.969] | 0.416 [0.253, 0.920] |
| eurosat | 3 | metric-privacy | oracle_noise_level | 0.699 [0.527, 0.871] | 0.202 [0.036, 0.604] | 0.013 [0.004, 0.453] |
| eurosat | 3 | metric-privacy | increment | 0.929 [0.834, 0.996] | 0.742 [0.556, 0.989] | 0.696 [0.504, 0.989] |
| eurosat | 3 | metric-privacy | loss | 0.803 [0.669, 0.992] | 0.062 [0.062, 0.960] | 0.036 [0.036, 0.882] |
| eurosat | 5 | global-dp | noise_level | 0.947 [0.927, 0.967] | 0.879 [0.805, 0.939] | 0.271 [0.025, 0.767] |
| eurosat | 5 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 5 | global-dp | increment | 0.799 [0.715, 0.920] | 0.448 [0.256, 0.724] | 0.399 [0.221, 0.675] |
| eurosat | 5 | global-dp | loss | 0.696 [0.645, 0.803] | 0.128 [0.099, 0.231] | 0.081 [0.032, 0.147] |
| eurosat | 5 | metric-privacy | noise_level | 0.792 [0.710, 0.887] | 0.096 [0.079, 0.516] | 0.067 [0.021, 0.229] |
| eurosat | 5 | metric-privacy | oracle_noise_level | 0.665 [0.559, 0.778] | 0.047 [0.041, 0.176] | 0.012 [0.001, 0.040] |
| eurosat | 5 | metric-privacy | increment | 0.785 [0.707, 0.906] | 0.460 [0.256, 0.721] | 0.401 [0.232, 0.688] |
| eurosat | 5 | metric-privacy | loss | 0.695 [0.643, 0.804] | 0.137 [0.097, 0.231] | 0.072 [0.029, 0.137] |
| eurosat | 8 | global-dp | noise_level | 0.771 [0.737, 0.817] | 0.237 [0.095, 0.347] | 0.018 [0.007, 0.043] |
| eurosat | 8 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 8 | global-dp | increment | 0.669 [0.620, 0.751] | 0.084 [0.058, 0.348] | 0.081 [0.025, 0.237] |
| eurosat | 8 | global-dp | loss | 0.771 [0.729, 0.832] | 0.159 [0.117, 0.384] | 0.069 [0.049, 0.171] |
| eurosat | 8 | metric-privacy | noise_level | 0.615 [0.548, 0.695] | 0.102 [0.059, 0.212] | 0.013 [0.009, 0.094] |
| eurosat | 8 | metric-privacy | oracle_noise_level | 0.572 [0.503, 0.652] | 0.066 [0.039, 0.158] | 0.010 [0.006, 0.046] |
| eurosat | 8 | metric-privacy | increment | 0.697 [0.645, 0.787] | 0.083 [0.083, 0.426] | 0.083 [0.041, 0.248] |
| eurosat | 8 | metric-privacy | loss | 0.778 [0.735, 0.838] | 0.169 [0.118, 0.417] | 0.068 [0.046, 0.184] |
| alzheimer | 3 | global-dp | noise_level | 0.984 [0.980, 0.988] | 0.931 [0.882, 0.958] | 0.818 [0.731, 0.893] |
| alzheimer | 3 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 3 | global-dp | increment | 0.711 [0.618, 0.802] | 0.151 [0.064, 0.467] | 0.049 [0.020, 0.262] |
| alzheimer | 3 | global-dp | loss | 0.892 [0.814, 0.996] | 0.249 [0.187, 0.973] | 0.164 [0.084, 0.880] |
| alzheimer | 3 | metric-privacy | noise_level | 0.641 [0.489, 0.798] | 0.042 [0.000, 0.387] | 0.007 [0.000, 0.084] |
| alzheimer | 3 | metric-privacy | oracle_noise_level | 0.634 [0.483, 0.793] | 0.038 [0.000, 0.376] | 0.004 [0.000, 0.080] |
| alzheimer | 3 | metric-privacy | increment | 0.623 [0.553, 0.717] | 0.064 [0.000, 0.171] | 0.000 [0.000, 0.082] |
| alzheimer | 3 | metric-privacy | loss | 0.864 [0.782, 0.995] | 0.291 [0.178, 0.967] | 0.111 [0.087, 0.887] |
| alzheimer | 5 | global-dp | noise_level | 0.707 [0.666, 0.751] | 0.171 [0.135, 0.236] | 0.052 [0.031, 0.099] |
| alzheimer | 5 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 5 | global-dp | increment | 0.581 [0.479, 0.682] | 0.053 [0.021, 0.125] | 0.004 [0.000, 0.031] |
| alzheimer | 5 | global-dp | loss | 0.801 [0.725, 0.890] | 0.137 [0.083, 0.544] | 0.067 [0.032, 0.293] |
| alzheimer | 5 | metric-privacy | noise_level | 0.632 [0.552, 0.718] | 0.144 [0.053, 0.279] | 0.047 [0.001, 0.149] |
| alzheimer | 5 | metric-privacy | oracle_noise_level | 0.630 [0.550, 0.716] | 0.144 [0.053, 0.279] | 0.045 [0.001, 0.147] |
| alzheimer | 5 | metric-privacy | increment | 0.623 [0.534, 0.717] | 0.057 [0.012, 0.317] | 0.011 [0.000, 0.237] |
| alzheimer | 5 | metric-privacy | loss | 0.806 [0.723, 0.900] | 0.161 [0.093, 0.492] | 0.116 [0.044, 0.292] |
| alzheimer | 8 | global-dp | noise_level | 0.553 [0.532, 0.572] | 0.068 [0.030, 0.093] | 0.003 [0.001, 0.021] |
| alzheimer | 8 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 8 | global-dp | increment | 0.544 [0.451, 0.639] | 0.114 [0.012, 0.186] | 0.009 [0.000, 0.089] |
| alzheimer | 8 | global-dp | loss | 0.721 [0.679, 0.777] | 0.215 [0.137, 0.338] | 0.077 [0.037, 0.183] |
| alzheimer | 8 | metric-privacy | noise_level | 0.559 [0.483, 0.640] | 0.039 [0.012, 0.092] | 0.003 [0.000, 0.025] |
| alzheimer | 8 | metric-privacy | oracle_noise_level | 0.558 [0.482, 0.639] | 0.040 [0.012, 0.092] | 0.002 [0.000, 0.025] |
| alzheimer | 8 | metric-privacy | increment | 0.588 [0.495, 0.693] | 0.102 [0.031, 0.193] | 0.042 [0.000, 0.112] |
| alzheimer | 8 | metric-privacy | loss | 0.691 [0.654, 0.745] | 0.250 [0.145, 0.358] | 0.138 [0.067, 0.256] |

### Attacks per cell: noisy view, top-decile targets

AUC; TPR at 5% FPR; TPR at 1% FPR; value [95% cluster-bootstrap interval].

| Dataset | Clients | Mechanism | Attacker | AUC | TPR@5% | TPR@1% |
|---|---|---|---|---|---|---|
| eurosat | 3 | global-dp | noise_level | 0.982 [0.967, 1.000] | 0.953 [0.907, 1.000] | 0.907 [0.827, 1.000] |
| eurosat | 3 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 3 | global-dp | increment | 0.788 [0.713, 1.000] | 0.080 [0.073, 1.000] | 0.033 [0.033, 1.000] |
| eurosat | 3 | global-dp | loss | 0.620 [0.320, 0.975] | 0.000 [0.000, 0.760] | 0.000 [0.000, 0.620] |
| eurosat | 3 | metric-privacy | noise_level | 0.996 [0.983, 1.000] | 0.967 [0.920, 1.000] | 0.960 [0.900, 1.000] |
| eurosat | 3 | metric-privacy | oracle_noise_level | 0.989 [0.974, 1.000] | 0.907 [0.853, 1.000] | 0.833 [0.707, 1.000] |
| eurosat | 3 | metric-privacy | increment | 0.717 [0.470, 1.000] | 0.000 [0.000, 1.000] | 0.000 [0.000, 1.000] |
| eurosat | 3 | metric-privacy | loss | 0.610 [0.455, 0.903] | 0.000 [0.000, 0.540] | 0.000 [0.000, 0.500] |
| eurosat | 5 | global-dp | noise_level | 0.918 [0.842, 0.964] | 0.767 [0.340, 0.940] | 0.160 [0.027, 0.920] |
| eurosat | 5 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 5 | global-dp | increment | 0.503 [0.220, 0.800] | 0.247 [0.000, 0.247] | 0.040 [0.000, 0.100] |
| eurosat | 5 | global-dp | loss | 0.600 [0.594, 0.838] | 0.260 [0.047, 0.480] | 0.140 [0.000, 0.260] |
| eurosat | 5 | metric-privacy | noise_level | 0.994 [0.991, 1.000] | 0.973 [0.947, 1.000] | 0.973 [0.947, 1.000] |
| eurosat | 5 | metric-privacy | oracle_noise_level | 0.997 [0.995, 1.000] | 0.987 [0.973, 1.000] | 0.967 [0.933, 1.000] |
| eurosat | 5 | metric-privacy | increment | 0.552 [0.398, 0.896] | 0.300 [0.000, 0.820] | 0.240 [0.000, 0.540] |
| eurosat | 5 | metric-privacy | loss | 0.606 [0.530, 0.872] | 0.220 [0.060, 0.247] | 0.020 [0.013, 0.060] |
| eurosat | 8 | global-dp | noise_level | 0.690 [0.690, 0.849] | 0.027 [0.013, 0.620] | 0.000 [0.000, 0.540] |
| eurosat | 8 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 8 | global-dp | increment | 0.456 [0.338, 0.574] | 0.073 [0.000, 0.460] | 0.000 [0.000, 0.220] |
| eurosat | 8 | global-dp | loss | 0.487 [0.377, 0.626] | 0.047 [0.013, 0.240] | 0.000 [0.000, 0.100] |
| eurosat | 8 | metric-privacy | noise_level | 0.891 [0.817, 1.000] | 0.687 [0.373, 1.000] | 0.520 [0.227, 1.000] |
| eurosat | 8 | metric-privacy | oracle_noise_level | 0.887 [0.829, 1.000] | 0.687 [0.373, 1.000] | 0.553 [0.233, 1.000] |
| eurosat | 8 | metric-privacy | increment | 0.547 [0.522, 0.702] | 0.193 [0.000, 0.387] | 0.100 [0.000, 0.300] |
| eurosat | 8 | metric-privacy | loss | 0.446 [0.272, 0.820] | 0.013 [0.000, 0.200] | 0.000 [0.000, 0.020] |
| alzheimer | 3 | global-dp | noise_level | 0.983 [0.969, 0.989] | 0.907 [0.820, 0.967] | 0.773 [0.700, 0.960] |
| alzheimer | 3 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 3 | global-dp | increment | 0.698 [0.577, 0.892] | 0.220 [0.100, 0.640] | 0.120 [0.000, 0.620] |
| alzheimer | 3 | global-dp | loss | 0.928 [0.875, 1.000] | 0.720 [0.360, 1.000] | 0.407 [0.220, 1.000] |
| alzheimer | 3 | metric-privacy | noise_level | 0.791 [0.742, 0.970] | 0.413 [0.240, 0.860] | 0.260 [0.060, 0.520] |
| alzheimer | 3 | metric-privacy | oracle_noise_level | 0.787 [0.738, 0.969] | 0.407 [0.220, 0.840] | 0.253 [0.060, 0.520] |
| alzheimer | 3 | metric-privacy | increment | 0.633 [0.609, 0.674] | 0.160 [0.100, 0.460] | 0.067 [0.067, 0.460] |
| alzheimer | 3 | metric-privacy | loss | 0.964 [0.952, 0.988] | 0.833 [0.753, 0.940] | 0.633 [0.520, 0.840] |
| alzheimer | 5 | global-dp | noise_level | 0.706 [0.698, 0.722] | 0.200 [0.120, 0.220] | 0.060 [0.020, 0.087] |
| alzheimer | 5 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 5 | global-dp | increment | 0.497 [0.367, 0.615] | 0.040 [0.040, 0.280] | 0.040 [0.040, 0.280] |
| alzheimer | 5 | global-dp | loss | 0.662 [0.623, 0.866] | 0.293 [0.180, 0.540] | 0.093 [0.013, 0.220] |
| alzheimer | 5 | metric-privacy | noise_level | 0.850 [0.824, 0.990] | 0.580 [0.313, 0.980] | 0.547 [0.307, 0.900] |
| alzheimer | 5 | metric-privacy | oracle_noise_level | 0.850 [0.824, 0.990] | 0.580 [0.313, 0.980] | 0.553 [0.307, 0.900] |
| alzheimer | 5 | metric-privacy | increment | 0.656 [0.452, 0.878] | 0.153 [0.000, 0.560] | 0.040 [0.000, 0.560] |
| alzheimer | 5 | metric-privacy | loss | 0.668 [0.608, 0.957] | 0.280 [0.060, 0.720] | 0.100 [0.040, 0.680] |
| alzheimer | 8 | global-dp | noise_level | 0.531 [0.511, 0.570] | 0.060 [0.020, 0.067] | 0.007 [0.000, 0.020] |
| alzheimer | 8 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 8 | global-dp | increment | 0.515 [0.347, 0.686] | 0.113 [0.000, 0.420] | 0.093 [0.000, 0.400] |
| alzheimer | 8 | global-dp | loss | 0.579 [0.534, 0.767] | 0.080 [0.080, 0.280] | 0.013 [0.000, 0.240] |
| alzheimer | 8 | metric-privacy | noise_level | 0.795 [0.671, 0.990] | 0.360 [0.200, 1.000] | 0.267 [0.060, 0.720] |
| alzheimer | 8 | metric-privacy | oracle_noise_level | 0.795 [0.670, 0.990] | 0.360 [0.200, 1.000] | 0.267 [0.060, 0.720] |
| alzheimer | 8 | metric-privacy | increment | 0.636 [0.592, 0.740] | 0.227 [0.113, 0.540] | 0.173 [0.067, 0.500] |
| alzheimer | 8 | metric-privacy | loss | 0.603 [0.571, 0.742] | 0.100 [0.100, 0.320] | 0.047 [0.020, 0.180] |

### Attacks per cell: noisy view, all targets

AUC; TPR at 5% FPR; TPR at 1% FPR; value [95% cluster-bootstrap interval].

| Dataset | Clients | Mechanism | Attacker | AUC | TPR@5% | TPR@1% |
|---|---|---|---|---|---|---|
| eurosat | 3 | global-dp | noise_level | 0.991 [0.981, 0.999] | 0.976 [0.949, 0.998] | 0.924 [0.889, 0.982] |
| eurosat | 3 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 3 | global-dp | increment | 0.734 [0.644, 0.887] | 0.122 [0.111, 0.533] | 0.118 [0.027, 0.422] |
| eurosat | 3 | global-dp | loss | 0.619 [0.520, 0.789] | 0.096 [0.000, 0.293] | 0.000 [0.000, 0.287] |
| eurosat | 3 | metric-privacy | noise_level | 0.951 [0.912, 0.988] | 0.820 [0.571, 0.969] | 0.416 [0.253, 0.920] |
| eurosat | 3 | metric-privacy | oracle_noise_level | 0.699 [0.527, 0.871] | 0.202 [0.036, 0.604] | 0.013 [0.004, 0.453] |
| eurosat | 3 | metric-privacy | increment | 0.691 [0.611, 0.846] | 0.133 [0.018, 0.389] | 0.107 [0.000, 0.347] |
| eurosat | 3 | metric-privacy | loss | 0.624 [0.525, 0.802] | 0.096 [0.000, 0.287] | 0.002 [0.000, 0.280] |
| eurosat | 5 | global-dp | noise_level | 0.947 [0.927, 0.967] | 0.879 [0.805, 0.939] | 0.271 [0.025, 0.767] |
| eurosat | 5 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 5 | global-dp | increment | 0.570 [0.500, 0.658] | 0.112 [0.000, 0.247] | 0.063 [0.000, 0.104] |
| eurosat | 5 | global-dp | loss | 0.574 [0.516, 0.664] | 0.057 [0.011, 0.125] | 0.021 [0.000, 0.043] |
| eurosat | 5 | metric-privacy | noise_level | 0.792 [0.710, 0.887] | 0.096 [0.079, 0.516] | 0.067 [0.021, 0.229] |
| eurosat | 5 | metric-privacy | oracle_noise_level | 0.665 [0.559, 0.778] | 0.047 [0.041, 0.176] | 0.012 [0.001, 0.040] |
| eurosat | 5 | metric-privacy | increment | 0.628 [0.559, 0.722] | 0.061 [0.011, 0.195] | 0.060 [0.000, 0.168] |
| eurosat | 5 | metric-privacy | loss | 0.559 [0.511, 0.632] | 0.044 [0.008, 0.113] | 0.000 [0.000, 0.048] |
| eurosat | 8 | global-dp | noise_level | 0.771 [0.737, 0.817] | 0.237 [0.095, 0.347] | 0.018 [0.007, 0.043] |
| eurosat | 8 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| eurosat | 8 | global-dp | increment | 0.593 [0.538, 0.663] | 0.068 [0.026, 0.163] | 0.042 [0.001, 0.112] |
| eurosat | 8 | global-dp | loss | 0.553 [0.519, 0.610] | 0.047 [0.032, 0.102] | 0.021 [0.007, 0.048] |
| eurosat | 8 | metric-privacy | noise_level | 0.615 [0.548, 0.695] | 0.102 [0.059, 0.212] | 0.013 [0.009, 0.094] |
| eurosat | 8 | metric-privacy | oracle_noise_level | 0.572 [0.503, 0.652] | 0.066 [0.039, 0.158] | 0.010 [0.006, 0.046] |
| eurosat | 8 | metric-privacy | increment | 0.593 [0.549, 0.655] | 0.057 [0.025, 0.118] | 0.024 [0.000, 0.059] |
| eurosat | 8 | metric-privacy | loss | 0.542 [0.512, 0.595] | 0.049 [0.030, 0.103] | 0.014 [0.011, 0.025] |
| alzheimer | 3 | global-dp | noise_level | 0.984 [0.980, 0.988] | 0.931 [0.882, 0.958] | 0.818 [0.731, 0.893] |
| alzheimer | 3 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 3 | global-dp | increment | 0.666 [0.588, 0.751] | 0.236 [0.027, 0.296] | 0.009 [0.009, 0.253] |
| alzheimer | 3 | global-dp | loss | 0.896 [0.808, 0.968] | 0.449 [0.211, 0.802] | 0.198 [0.073, 0.589] |
| alzheimer | 3 | metric-privacy | noise_level | 0.641 [0.489, 0.798] | 0.042 [0.000, 0.387] | 0.007 [0.000, 0.084] |
| alzheimer | 3 | metric-privacy | oracle_noise_level | 0.634 [0.483, 0.793] | 0.038 [0.000, 0.376] | 0.004 [0.000, 0.080] |
| alzheimer | 3 | metric-privacy | increment | 0.649 [0.566, 0.769] | 0.087 [0.020, 0.280] | 0.011 [0.000, 0.093] |
| alzheimer | 3 | metric-privacy | loss | 0.865 [0.768, 0.979] | 0.249 [0.147, 0.882] | 0.176 [0.067, 0.820] |
| alzheimer | 5 | global-dp | noise_level | 0.707 [0.666, 0.751] | 0.171 [0.135, 0.236] | 0.052 [0.031, 0.099] |
| alzheimer | 5 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 5 | global-dp | increment | 0.549 [0.490, 0.617] | 0.069 [0.040, 0.136] | 0.015 [0.008, 0.037] |
| alzheimer | 5 | global-dp | loss | 0.693 [0.633, 0.779] | 0.089 [0.067, 0.248] | 0.040 [0.019, 0.064] |
| alzheimer | 5 | metric-privacy | noise_level | 0.632 [0.552, 0.718] | 0.144 [0.053, 0.279] | 0.047 [0.001, 0.149] |
| alzheimer | 5 | metric-privacy | oracle_noise_level | 0.630 [0.550, 0.716] | 0.144 [0.053, 0.279] | 0.045 [0.001, 0.147] |
| alzheimer | 5 | metric-privacy | increment | 0.524 [0.388, 0.647] | 0.115 [0.051, 0.207] | 0.016 [0.012, 0.105] |
| alzheimer | 5 | metric-privacy | loss | 0.755 [0.667, 0.873] | 0.093 [0.056, 0.491] | 0.029 [0.015, 0.234] |
| alzheimer | 8 | global-dp | noise_level | 0.553 [0.532, 0.572] | 0.068 [0.030, 0.093] | 0.003 [0.001, 0.021] |
| alzheimer | 8 | global-dp | oracle_noise_level | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| alzheimer | 8 | global-dp | increment | 0.538 [0.488, 0.589] | 0.075 [0.034, 0.142] | 0.010 [0.005, 0.056] |
| alzheimer | 8 | global-dp | loss | 0.672 [0.620, 0.733] | 0.093 [0.058, 0.159] | 0.019 [0.011, 0.041] |
| alzheimer | 8 | metric-privacy | noise_level | 0.559 [0.483, 0.640] | 0.039 [0.012, 0.092] | 0.003 [0.000, 0.025] |
| alzheimer | 8 | metric-privacy | oracle_noise_level | 0.558 [0.482, 0.639] | 0.040 [0.012, 0.092] | 0.002 [0.000, 0.025] |
| alzheimer | 8 | metric-privacy | increment | 0.566 [0.480, 0.655] | 0.066 [0.044, 0.142] | 0.015 [0.008, 0.043] |
| alzheimer | 8 | metric-privacy | loss | 0.669 [0.615, 0.735] | 0.135 [0.058, 0.228] | 0.019 [0.012, 0.123] |

### Top-decile targets

The largest client by local dataset size in each federation and seed (`ceil(0.1 n)` = 1 here). The partition depends only on dataset, client count and seed, so both mechanisms share it.

| Dataset | Clients | Seed | Top-decile target | Its records | Records of every client |
|---|---|---|---|---|---|
| alzheimer | 3 | 42 | 2 | 2,666 | 1,121, 1,333, 2,666 |
| alzheimer | 3 | 43 | 0 | 2,350 | 2,350, 1,975, 795 |
| alzheimer | 3 | 44 | 1 | 2,470 | 2,151, 2,470, 499 |
| alzheimer | 5 | 42 | 1 | 2,328 | 302, 2,328, 1,622, 138, 730 |
| alzheimer | 5 | 43 | 2 | 2,340 | 471, 1,258, 2,340, 597, 454 |
| alzheimer | 5 | 44 | 1 | 2,035 | 176, 2,035, 822, 910, 1,177 |
| alzheimer | 8 | 42 | 5 | 1,891 | 236, 253, 407, 307, 440, 1,891, 1,040, 546 |
| alzheimer | 8 | 43 | 5 | 1,752 | 749, 367, 275, 265, 454, 1,752, 128, 1,130 |
| alzheimer | 8 | 44 | 4 | 1,288 | 770, 1,026, 283, 504, 1,288, 584, 258, 407 |
| eurosat | 3 | 42 | 1 | 9,573 | 3,582, 9,573, 8,445 |
| eurosat | 3 | 43 | 2 | 11,774 | 3,047, 6,779, 11,774 |
| eurosat | 3 | 44 | 2 | 10,122 | 3,588, 7,890, 10,122 |
| eurosat | 5 | 42 | 4 | 7,984 | 4,622, 2,633, 2,521, 3,840, 7,984 |
| eurosat | 5 | 43 | 4 | 6,906 | 3,265, 3,454, 4,272, 3,703, 6,906 |
| eurosat | 5 | 44 | 0 | 5,981 | 5,981, 5,542, 3,941, 2,976, 3,160 |
| eurosat | 8 | 42 | 7 | 4,699 | 3,525, 2,808, 1,724, 1,848, 1,660, 2,893, 2,443, 4,699 |
| eurosat | 8 | 43 | 7 | 4,100 | 2,194, 2,620, 3,034, 2,884, 3,097, 1,503, 2,168, 4,100 |
| eurosat | 8 | 44 | 2 | 4,715 | 1,903, 1,832, 4,715, 1,516, 2,660, 2,842, 2,913, 3,219 |

### Server test accuracy at round 50, every trajectory

IN = all clients; OUT-t = the federation without client t. Accuracy as a fraction.


**eurosat**

| Clients | Mechanism | Seed | IN | OUT-0 | OUT-1 | OUT-2 | OUT-3 | OUT-4 | OUT-5 | OUT-6 | OUT-7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | global-dp | 42 | 0.9333 | 0.9185 | 0.7844 | 0.8630 |  |  |  |  |  |
| 3 | global-dp | 43 | 0.9296 | 0.8474 | 0.8385 | 0.8467 |  |  |  |  |  |
| 3 | global-dp | 44 | 0.9215 | 0.9481 | 0.8593 | 0.8148 |  |  |  |  |  |
| 3 | metric-privacy | 42 | 0.9370 | 0.9244 | 0.8319 | 0.8733 |  |  |  |  |  |
| 3 | metric-privacy | 43 | 0.9259 | 0.8415 | 0.8437 | 0.8570 |  |  |  |  |  |
| 3 | metric-privacy | 44 | 0.9104 | 0.9496 | 0.8459 | 0.8267 |  |  |  |  |  |
| 5 | global-dp | 42 | 0.9096 | 0.9259 | 0.9081 | 0.8681 | 0.8904 | 0.8748 |  |  |  |
| 5 | global-dp | 43 | 0.9356 | 0.9126 | 0.9193 | 0.9081 | 0.9407 | 0.8844 |  |  |  |
| 5 | global-dp | 44 | 0.8778 | 0.8148 | 0.9007 | 0.8667 | 0.8963 | 0.8763 |  |  |  |
| 5 | metric-privacy | 42 | 0.9119 | 0.9267 | 0.9163 | 0.8748 | 0.8948 | 0.8637 |  |  |  |
| 5 | metric-privacy | 43 | 0.9341 | 0.9141 | 0.9074 | 0.9119 | 0.9393 | 0.8859 |  |  |  |
| 5 | metric-privacy | 44 | 0.8874 | 0.8163 | 0.8889 | 0.8674 | 0.8948 | 0.8719 |  |  |  |
| 8 | global-dp | 42 | 0.9333 | 0.9007 | 0.9267 | 0.9193 | 0.9133 | 0.9289 | 0.9304 | 0.9244 | 0.9081 |
| 8 | global-dp | 43 | 0.8963 | 0.8822 | 0.8993 | 0.9081 | 0.8993 | 0.8844 | 0.9156 | 0.9244 | 0.9052 |
| 8 | global-dp | 44 | 0.9059 | 0.8748 | 0.8889 | 0.9007 | 0.9237 | 0.8985 | 0.9000 | 0.9252 | 0.8933 |
| 8 | metric-privacy | 42 | 0.9267 | 0.9052 | 0.9207 | 0.9163 | 0.9185 | 0.9207 | 0.9230 | 0.9185 | 0.9141 |
| 8 | metric-privacy | 43 | 0.9059 | 0.8681 | 0.9059 | 0.9193 | 0.8800 | 0.8719 | 0.9096 | 0.9133 | 0.8978 |
| 8 | metric-privacy | 44 | 0.8993 | 0.8770 | 0.8919 | 0.8859 | 0.9230 | 0.8874 | 0.9037 | 0.9237 | 0.8785 |

**alzheimer**

| Clients | Mechanism | Seed | IN | OUT-0 | OUT-1 | OUT-2 | OUT-3 | OUT-4 | OUT-5 | OUT-6 | OUT-7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | global-dp | 42 | 0.8516 | 0.8297 | 0.7219 | 0.6797 |  |  |  |  |  |
| 3 | global-dp | 43 | 0.8625 | 0.7281 | 0.8172 | 0.8250 |  |  |  |  |  |
| 3 | global-dp | 44 | 0.8469 | 0.7422 | 0.7688 | 0.8187 |  |  |  |  |  |
| 3 | metric-privacy | 42 | 0.8781 | 0.8766 | 0.8172 | 0.7031 |  |  |  |  |  |
| 3 | metric-privacy | 43 | 0.8812 | 0.7250 | 0.8750 | 0.8781 |  |  |  |  |  |
| 3 | metric-privacy | 44 | 0.8438 | 0.7516 | 0.8000 | 0.8047 |  |  |  |  |  |
| 5 | global-dp | 42 | 0.7594 | 0.8422 | 0.6828 | 0.7531 | 0.7734 | 0.6687 |  |  |  |
| 5 | global-dp | 43 | 0.7422 | 0.5437 | 0.5797 | 0.6422 | 0.7406 | 0.7063 |  |  |  |
| 5 | global-dp | 44 | 0.8609 | 0.8313 | 0.6984 | 0.7500 | 0.7406 | 0.7750 |  |  |  |
| 5 | metric-privacy | 42 | 0.8406 | 0.8812 | 0.7578 | 0.7359 | 0.8359 | 0.7844 |  |  |  |
| 5 | metric-privacy | 43 | 0.7562 | 0.6516 | 0.5125 | 0.7016 | 0.7641 | 0.6875 |  |  |  |
| 5 | metric-privacy | 44 | 0.8516 | 0.8859 | 0.6828 | 0.7781 | 0.7391 | 0.8219 |  |  |  |
| 8 | global-dp | 42 | 0.8297 | 0.7891 | 0.8438 | 0.8031 | 0.7500 | 0.7031 | 0.7250 | 0.7766 | 0.7078 |
| 8 | global-dp | 43 | 0.8141 | 0.6984 | 0.8016 | 0.7812 | 0.7531 | 0.7828 | 0.7656 | 0.7250 | 0.7844 |
| 8 | global-dp | 44 | 0.6281 | 0.7531 | 0.7328 | 0.7328 | 0.7234 | 0.6344 | 0.6344 | 0.6609 | 0.7188 |
| 8 | metric-privacy | 42 | 0.8141 | 0.8297 | 0.8547 | 0.8297 | 0.8250 | 0.8125 | 0.7297 | 0.7937 | 0.7781 |
| 8 | metric-privacy | 43 | 0.8562 | 0.7188 | 0.8094 | 0.8156 | 0.8516 | 0.8125 | 0.7328 | 0.7625 | 0.8078 |
| 8 | metric-privacy | 44 | 0.7016 | 0.7625 | 0.7016 | 0.7906 | 0.6828 | 0.6719 | 0.7156 | 0.6969 | 0.7531 |
