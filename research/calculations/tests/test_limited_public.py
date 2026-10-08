import numpy as np
from research.calculations.limited_public_probe import noise_total, step, UNCLIPPED
from research.calculations.public_residual_probe import bounded_public, public_scale, random_shares


def test_clipping_bound_and_unclipped_identity():
    q=np.random.default_rng(3).normal(size=(8,12))*5
    signal,f=bounded_public(q,.1); assert np.all(np.linalg.norm(signal,axis=1)<=.1/8+1e-12)
    unclipped,f=bounded_public(q,UNCLIPPED); assert np.allclose(unclipped,q/8) and np.all(f==1)


def test_peer_noise_convention_and_scale():
    sigma=public_scale(.03,.65); n=noise_total(20000,5,6,sigma)
    assert np.allclose(n,random_shares(20000,5)[:,:,:6].sum(1)*sigma/np.sqrt(7))
    assert abs(n.var(0).mean()/(sigma**2*8/7)-1)<.05


def test_step_zero_signal_is_public_offset_only():
    b=np.zeros((3,17)); off=np.ones((3,17)); dec=np.eye(51)[:2]
    out=step(b,.5,off,np.zeros(2),dec); assert out.shape==(1,3,17) and np.allclose(out[0],-.5*off)


def test_stacked_step_is_bounded_by_eta_times_cap_and_noise_free_is_deterministic():
    from research.calculations.stacked_constructor_probe import stacked_theta
    from research.calculations.public_residual_probe import public_hessian
    rng=np.random.default_rng(11); x=rng.normal(size=(64,17)); control=rng.normal(size=(3,17))*.1; h=public_hessian(x,control)
    query=rng.normal(size=(8,51))*3; cfg={'dimension':12,'cap':.05,'eta':7.,'risk':.65}
    a=stacked_theta(control,h,query,cfg,noiseless=True); b=stacked_theta(control,h,query,cfg,noiseless=True)
    assert np.array_equal(a,b) and np.linalg.norm(a[0]-control)<=cfg['eta']*cfg['cap']+1e-12
    noisy=stacked_theta(control,h,query,cfg,draws=16,seed=3); assert noisy.shape==(16,3,17)


def test_radial_total_matches_earlier_matched_cia_sampler_and_variance():
    from research.calculations.noise_law_probe import radial_total
    from research.calculations.matched_cia_probe import noise_objects, radial_scale_ratio
    d,risk,cap=12,.65,.01; scale=(cap/8)*radial_scale_ratio(d,risk)
    ref=noise_objects(np.zeros((8,d)),'radial',scale,64,5).sum(1)
    assert np.allclose(radial_total(64,5,d,cap,risk),ref)
    big=radial_total(40000,9,d,cap,risk); assert abs(big.var(0).mean()/(8/7*(d+1)*scale**2)-1)<.05
