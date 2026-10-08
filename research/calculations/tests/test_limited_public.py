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
