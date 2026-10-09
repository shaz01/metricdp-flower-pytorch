"""Two-round bounded analytical comparison, not FL training or CIA evaluation.

uv run python research/calculations/protected_history_two_round.py
"""
from __future__ import annotations
import itertools
import json
import math
from pathlib import Path
import numpy as np

S = 0.8


def box_minimum(matrix, rhs, constant, cap=S):
    """Enumerate box faces of a strictly convex quadratic in dimension <=3."""
    best = (math.inf, None)
    for status in itertools.product((-1, 0, 1), repeat=len(rhs)):
        free = [i for i, value in enumerate(status) if value == -1]
        fixed = [i for i, value in enumerate(status) if value != -1]
        x = np.array([cap if value == 1 else 0.0 for value in status])
        if free:
            x[free] = np.linalg.solve(matrix[np.ix_(free, free)],
                                     rhs[free] - matrix[np.ix_(free, fixed)] @ x[fixed])
        if np.any(x < -1e-10) or np.any(x > cap + 1e-10):
            continue
        value = float(x @ matrix @ x - 2 * rhs @ x + constant)
        if value < best[0]:
            best = value, x
    assert best[1] is not None
    return best


def ordered_minimum(matrix, rhs, constant):
    # Convex optimum either has L>S or lies on the L=S face.
    interior = box_minimum(matrix, rhs, constant)
    if interior[1][1] >= interior[1][2] - 1e-12:
        return interior
    transform = np.array([[1., 0.], [0., 1.], [0., 1.]])
    value, reduced = box_minimum(transform.T @ matrix @ transform,
                                 transform.T @ rhs, constant)
    return value, transform @ reduced


def quadratic(total, probe_budget, clients, adaptive):
    second_budget = total - probe_budget
    q = 0.5 * (1 + probe_budget / 2) * math.exp(-probe_budget / 2)
    p = 1 - q
    if adaptive:
        matrix = np.array([[1+16/probe_budget**2, (1+p)/2, q/2],
                           [(1+p)/2, p+8/second_budget**2, 0],
                           [q/2, 0, q+8/second_budget**2]])
        center = np.array([1., p, q])
    else:
        matrix = np.array([[1+16/probe_budget**2, 1],
                           [1, 1+16/second_budget**2]])
        center = np.ones(2)
    # Four times the iid federation-average risk, including its nonzero mean bias.
    factor = (clients - 1) / (2 * clients)
    matrix = matrix / clients + factor * np.outer(center, center)
    rhs = 2 * S * center / clients + factor * 2 * S * center
    constant = 4 * S*S / clients + factor * 4 * S*S
    return matrix, rhs, constant, q


def optimize(total, clients, adaptive, intervals):
    # Explicit valid boundary: one public zero upload, all budget in the other.
    weight = (clients+1)/(2*clients)
    radius = min(S, 2*weight*S/(weight+16/(clients*total**2)))
    risk = weight*(radius/2-S)**2+4*radius**2/(clients*total**2)
    best = {"risk":risk,"probe_epsilon":total,"second_epsilon":0.0,
            "probe_radius":radius,"profile_radii":[0.0,0.0] if adaptive else [0.0],
            "wrong_profile_probability":None,"mode":"nonadaptive_zero_second_round"}
    for j in range(1, intervals):
        e0 = total*j/intervals
        matrix, rhs, constant, q = quadratic(total, e0, clients, adaptive)
        value, geometry = (ordered_minimum if adaptive else box_minimum)(matrix,rhs,constant)
        if value/4 < best["risk"]:
            assert geometry[0] > 0  # q formula assumes a genuine nondegenerate probe.
            best = {"risk":value/4,"probe_epsilon":e0,"second_epsilon":total-e0,
                    "probe_radius":float(geometry[0]),"profile_radii":geometry[1:].tolist(),
                    "wrong_profile_probability":q if adaptive else None,"mode":"two_positive_budgets"}
    return best


def same_probe_control(total, clients, candidate):
    a0 = candidate["probe_radius"]
    e0 = candidate["probe_epsilon"]
    if candidate["second_epsilon"] == 0:
        return {"risk":candidate["risk"],"second_radius":0.0,"mode":"zero_second_round"}
    matrix, rhs, constant, _ = quadratic(total,e0,clients,False)
    a1 = float(np.clip((rhs[1]-matrix[1,0]*a0)/matrix[1,1],0,S))
    x = np.array([a0,a1])
    return {"risk":float((x@matrix@x-2*rhs@x+constant)/4),"second_radius":a1}


def one_release(total, clients):
    # Public diamond, all budget in one upload, directly estimate u (different schedule).
    bias_weight = (clients+1)/(2*clients)
    noise_weight = 16/(clients*total**2)
    radius = bias_weight*S/(bias_weight+noise_weight)
    return {"risk":bias_weight*(S-radius)**2+noise_weight*radius**2,
            "radius":radius,"scope":"different one-release schedule; not matched two-round arm"}


def validate():
    # Independent direct risk formula equals quadratic for both local and aggregate objectives.
    total,e0,a0,L,short = 8.,4.,.5,.6,.2
    e1=total-e0;q=.5*(1+e0/2)*math.exp(-e0/2);p=1-q
    r0=(a0-S)**2+16*a0*a0/e0**2
    r1=p*(L-S)**2+q*(short-S)**2+8*(L*L+short*short)/e1**2
    cross=(a0-S)*(p*(L-S)+q*(short-S))+a0*q*(L-short)/2
    local=(r0+r1+2*cross)/4
    for n in (1,8,48):
        expected=local/n+(n-1)/n*((a0+p*L+q*short-2*S)/2)**2/2
        matrix,rhs,constant,_=quadratic(total,e0,n,True)
        x=np.array([a0,L,short])
        assert math.isclose((x@matrix@x-2*rhs@x+constant)/4,expected,abs_tol=1e-12)
        # Equal profile radii must recover public law and its full risk.
        xa=np.array([a0,L,L]); xp=np.array([a0,L])
        mp,bp,kp,_=quadratic(total,e0,n,False)
        assert math.isclose(xa@matrix@xa-2*rhs@xa+constant,
                            xp@mp@xp-2*bp@xp+kp,abs_tol=1e-12)
    # Independent random-draw validation of selection probability and correlation term.
    rng=np.random.default_rng(20261004)
    z=rng.laplace(0,2*a0/e0,size=(400_000,2))
    wrong=np.abs(a0+z[:,0])<np.abs(z[:,1])
    samples=z[:,0]*wrong
    measured_q=float(wrong.mean()); measured_m=float(samples.mean())
    se_q=math.sqrt(q*(1-q)/len(z));se_m=float(samples.std(ddof=1)/math.sqrt(len(z)))
    assert abs(measured_q-q)<6*se_q
    assert abs(measured_m+a0*q/2)<6*se_m
    return {"sample_count":len(z),"seed":20261004,"expected_wrong":q,
            "measured_wrong":measured_q,"wrong_standard_error":se_q,
            "expected_cross_moment":-a0*q/2,"measured_cross_moment":measured_m,
            "cross_moment_standard_error":se_m,
            "scope":"sampler/formula sanity check; not a privacy certificate or CIA trial"}


def main():
    validation=validate()
    rows=[]
    for total in (1.,2.,4.,8.,16.):
        for n in (1,8,48):
            coarse=optimize(total,n,True,512)
            adaptive=optimize(total,n,True,1024)
            static=optimize(total,n,False,1024)
            same=same_probe_control(total,n,adaptive)
            single=one_release(total,n)
            rows.append({"epsilon_total":total,"clients":n,"adaptive":adaptive,
                         "optimized_static_two_round":static,"same_probe_static":same,
                         "one_release_public":single,
                         "public_population_mean_risk":S*S/(2*n),
                         "adaptive_grid_refinement_change":abs(coarse["risk"]-adaptive["risk"]),
                         "gain_vs_static_percent":100*(static["risk"]-adaptive["risk"])/static["risk"]})
    result={"kind":"two-round analytic risk optimization with bounded grids",
            "population":"iid updates .8e1 or .8e2, each probability .5; stationary across rounds",
            "released_estimator":"mean of two uploads per client; federation mean across clients",
            "privacy":"hypothetical full upload transcript; eta0+eta1=epsilon_total pure whole-input DP",
            "probe":"public isotropic diamond; independent Laplace coordinates",
            "construction":"argmax absolute probe coordinate; swapped diamond profiles; L>=S",
            "optimization":"box-constrained convex geometry at each budget grid; probe radius in(0,.8]",
            "checks":validation,"rows":rows,
            "limitations":["not a continuous-budget global optimum certificate",
                           "not optimization over arbitrary public bodies/estimators",
                           "stationary updates are not genuine two-round training dynamics",
                           "single-release control changes schedule",
                           "known population mean is an additional zero-budget public comparator",
                           "no CIA measurement, deep-model evidence or novelty claim"]}
    path=Path('results/client_specific_noise/protected_history_two_round.json')
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for row in rows:
        print(row['epsilon_total'],row['clients'],
              f"adaptive={row['adaptive']['risk']:.6f}",
              f"static={row['optimized_static_two_round']['risk']:.6f}",
              f"single={row['one_release_public']['risk']:.6f}",
              f"gain={row['gain_vs_static_percent']:.3f}%")


if __name__=='__main__':
    main()
