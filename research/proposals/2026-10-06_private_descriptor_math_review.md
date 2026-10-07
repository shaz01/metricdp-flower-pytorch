# Independent one-time descriptor and noise-law transfer review

2026-10-06. Read-only independent agents reviewed the two privacy contracts, descriptor centering and noise construction before utility inspection. Equations below are our transfer derivations unless expressly attributed. Scope: ideal real-arithmetic laws and a registered whole-dataset-contribution-versus-zero-dummy game; research selection/artifact release remains unaccounted.

## Gaussian aggregate versus individual channels

Let d_i(D_i) be a deterministic centered descriptor based only on own data and fixed public inputs. Define x_i=clip_S(a_i d_i), S=r/8, x_i(empty)=0. Weights are public and unchanged across neighboring worlds. Target change only affects x_T, with ||x_T||≤S.

For isotropic Gaussian noise whose unknown residual covariance isσ²I, the target dummy-edge zCDP cost isS²/(2σ²). With eight equal-variance already-weighted shares ofσ²/7, conditioning on the curious peer's data/mean/own coins leaves seven independent unknown shares with covarianceσ²I. Full aggregate variance is8σ²/7. Later Q/student training may depend only on the released aggregate and public information.

If all individual objects must support separate nonlinear Q_i, calibrate EACH object's noise atσ². Under a one-target dataset change, the joint independent object law has that same target-channel bound; budgets do not sum across unchanged clients. Its linear aggregate variance is8σ². Unweighting, simplex projection, softmax, pooling, sampling and training are postprocessing. Sevenfold variance is the cost of this sufficient stronger certificate, not an impossibility theorem for every final-only nonlinear protocol.

A target's private noise coins or scale must not be disclosed. Other clients' randomness is independent of target data; metadata/counts/weights/horizon are fixed. Continued raw-data refits, private early stopping or privately fitted public anchors invalidate the one-time postprocessing argument.

Analytic Gaussian uses the [Balle–Wang Theorem8](https://proceedings.mlr.press/v80/balle18a/balle18a.pdf) complete joint sensitivity condition. Independent bisection values atdelta1e-5 match across reviewers. Its(ε,δ) certificate is not the sameρ as the earlier zCDP conversion. Recalibrate sensitivity2S for replacement if the same(ε,δ) claim is needed.

## Radial law and divisible sampler — independent derivation

For d-dimensional radial noise with density f(z)=c_d exp(−||z||/s),

c_d=Γ(d/2)/(2π^(d/2)Γ(d)s^d).

For any target shiftv with||v||≤S, logf(z)−logf(z−v)≤||v||/s. Withs=S/epsilon this gives pureepsilon in both directions. Replacement has shift≤2S and pure2epsilon, or requires doubled calibration to retainepsilon. Whole dimensiond includes every anchor/contrast jointly.

Take alpha=(d+1)/2, G~Gamma(alpha,scale2s²) and Z|G~Normal(0,G I_d). Its characteristic function is(1+s²||t||²)^−alpha. Integrating the Gaussian mixture yields a modified-Bessel factor with orderalpha−d/2=1/2; the elementaryK_1/2 identity gives exactly the radial exponential density above. Thus coordinatevarianceE G=(d+1)s² and total energy d(d+1)s². The norm hasGamma(shape d,scale s), an independent sampler-validation reference.

Shares with G_i~Gamma(alpha/7,2s²) have characteristic functions whose product over seven unknown slots reconstructsalpha. The peer may remove only its own known share; seven unknown shares give the certified radial residual. Eight shares instead haveGamma shape8alpha/7 and coordinatevariance(8/7)(d+1)s², **not** the original radial exponential density. A matching central full aggregate law has this same mixture. A stronger central-only law can use fullalpha and variance(d+1)s².

For independently reusable client Q_i, use fullalpha at EACH client. These complete channels are individually pureepsilon, and subsequent joint learning is postprocessing. Gamma scales are hidden and shared across all coordinates within a vector; independent per-coordinateGamma draws produce a different law. For low-dimensional aggregate shares alpha/7<d/2, an individual share has a singular density atzero and no finite pure translation-DP guarantee for nonzero displacement. A cannot turn these low-noise objects into individually released Q_i.

## Numerical controls and limits

At S1,delta1e-5:

| Epsilon | Analytic Gaussian coordinate variance | Radial variance,d12 | Radial/Gaussian |
|---|---:|---:|---:|
|4|1.168910944858|0.812500000000|0.695091|
|8|0.360274939113|0.203125000000|0.563805|
|16|0.118458102286|0.050781250000|0.428686|

The radial law hasdelta0 and meets the common allowed(ε,delta1e-5) target. This is a regime/dimension comparison, not equivalence ofρ or proof of universal Laplace optimality. At dimension48 the variance penalty is larger; tail behavior, clipping and postprocessing can reverse utility ordering. Joint sampler moment/normalization checks and independent code/artifact review accompany the pilot. NumPy finite-precision/fixed-seed simulation is not an audited deployment mechanism; scientific publication of all arms is not a joint private release.
