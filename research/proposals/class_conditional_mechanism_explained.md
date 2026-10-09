# How public class-conditional residual protection works

This explains the bounded constructor in [the frozen protocol](2026-10-08_class_conditional_protocol.md). It is a research feasibility mechanism, not the final defense or a new noise-density claim.

## What clients construct

A labelled public dataset gives a common reference classifier B. At B, compute a gradient for each public example: the direction that would reduce its cross-entropy. Average those directions separately for each class; call them gamma_0,...,gamma_3. These are public reference directions, not private client statistics.

Client i computes its own class proportions p_i and its mean gradient g_i. Its predictable direction is sum_k p_ik gamma_k. For example, a client with80%class0 and20%class1 subtracts .8 gamma_0+.2 gamma_1. It then contributes the residual

    r_i = g_i - sum_k p_ik gamma_k.

The residual describes how this client's examples differ from public examples with the SAME label mix. Client proportions stay inside the client. The server does not receive proportions, class counts, or an unprotected residual. An empty registered client contributes zero, rather than a negative public reference.

The public encoder projects this residual into a predeclared subspace. The client clips the whole projected vector to a public cap C, divides by8, and adds a Gaussian share. Its conditional upload distribution is therefore

    U_i | D_i = Normal(clip_C(encode(r_i))/8, sigma² I / 7).

The center depends on the client's dataset. The covariance and cap are public and common; this phase does not estimate a unique private covariance for each client. Different client-specific centers alone do not constitute a novel Gaussian noise density. A future covariance or density adaptation needs its own complete-law privacy and absent-client treatment.

## What the server restores

The server sums the uploads and decodes the residual. It adds a public target-distribution gradient gamma_target before taking a step from B. In the baseline, target classes are equally weighted. If all clients are present, mean client proportions equal the target proportions, and clipping is inactive, the restored average is exactly the original average gradient, up to the explicitly retained public directions outside the projection.

The mechanism subtracts label-mixture-predictable directions without necessarily removing aggregate learning information. That is why it could allow a smaller clipping cap and less noise. It is a hypothesis to evaluate: norms can also shrink because useful private signal was removed, and clipping can invalidate aggregate equality.

## Why the absent-client world matters

When a client is replaced by an empty dummy, its protected residual disappears but the public restoration stays. The absent world consequently imputes a public class-conditional contribution. It is generally DIFFERENT from merely removing that client's raw gradient.

Writing Gamma p for sum_k p_k gamma_k, P for the public projection and a_i=1/8, the restored unbounded gradient for an active set A is

    P sum_{i in A} a_i g_i + (I-P) Gamma pi
      + P Gamma (pi - sum_{i in A} a_i p_i).

When all-IN mean proportions match pi, the last term is zero. Removing target T adds the imputation term a_T P Gamma p_T relative to raw removal. Thus the IN/OUT model shift is driven by the target residual, rather than its full raw gradient. This is a change to what learning information is released, not a free reduction of the privacy cost of the original raw query.

## Why exact class balancing is a separate control

A target-balanced client gradient first averages examples within each class, then weights those class means by the declared target prior pi. Its correct public reference is Gamma pi. Subtracting Gamma p_i from that query would mix different objectives. Missing private classes use their public class reference, while a completely empty client still outputs zero. This fallback can itself supply public information; controls must get the same public data.

Under unequal client sizes, fixed equal-client weights do not match pooled-record weights. After the prior stress, pooled class0 frequency is about.404 while mean client class0 proportion is about.318; the evaluation target.4 is separately specified. The constructor must not claim an unchanged aggregate objective under these conditions.

## What the Gaussian attack bound means

A curious peer subtracts its own upload and coins. Seven unknown shares remain, with total covariance sigma² I. The target's IN/OUT mean shift has norm at most C/8. Setting

    sigma = C / (8 sqrt(2) Phi^-1(q))

bounds the known-alternative optimal ROC AUC by q for the fixed query/settings. This is whole-dataset contribution versus empty among registered slots. It does not hide physical connection metadata or certify arbitrary dataset replacement. Full student parameters expose the affine released coordinates, so the optimal descriptor attack remains recoverable from the student plus peer state.

The central same-query control uses the same unknown sigma, but avoids the peer's known share inflating total training noise. Its training variance is sigma² rather than8 sigma²/7. That placement difference does not show a better noise density.

## What would make this a promising result

Lower norms are an initial diagnostic. The useful result would be lower fresh prediction loss than BOTH a strong public-only model and equally tuned raw/balanced/fixed-reference private controls, at matched conditional attack strength, across the three saved federations. Transferred prior and feature shifts test whether the construction depends on an accidentally matching public reference. The current public linear task may have too little useful private signal left after public training; failure should prompt a signal/headroom diagnosis before another density search.

See the phase findings for measured results. Offline tuning, public control selection on old development labels, saved operator diagnostics and finite precision are outside the modeled deployment guarantee. No population or novelty claim is inferred.
