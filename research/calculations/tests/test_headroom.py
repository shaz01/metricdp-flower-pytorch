import numpy as np
from research.calculations.headroom_probe import public_subset, development_halves, residual_moments, refine
from research.calculations.client_energy_filter_probe import C, scores


def test_nested_balanced_public_and_disjoint_development():
    ids=np.arange(512); labels=np.repeat(np.arange(4),128)
    a=public_subset(ids,labels,32,42); b=public_subset(ids,labels,128,42); c=public_subset(ids,labels,512,42)
    assert set(a)<set(b)<set(c) and np.array_equal(np.bincount(labels[a]),[8]*4)
    ids=np.arange(1024);labels=np.repeat(np.arange(4),256); s,t=development_halves(ids,labels)
    assert len(s)==len(t)==512 and not set(s)&set(t) and set(s)|set(t)==set(ids)


def test_residual_mean_variance_and_finetuning_descent():
    rng=np.random.default_rng(23); z=rng.normal(size=(20,3,17)); mean,var=residual_moments(z)
    assert np.allclose(mean,z.mean(0)) and np.allclose(var,np.trace(np.cov(z.reshape(20,51),rowvar=False))/20)
    x=rng.normal(size=(40,17));y=np.tile(np.arange(4),10);b=np.zeros((3,17));new=refine(x,y,b,1)
    assert scores(x,y,new[None])[0][0]<scores(x,y,b[None])[0][0]
    assert np.array_equal(b,np.zeros_like(b))


def test_fixed_class_count_variance_survives_public_centering():
    from research.calculations.headroom_probe import stratified_variance
    rng=np.random.default_rng(31); raw=rng.normal(size=(20,3,17));y=np.repeat(np.arange(4),5);gamma=rng.normal(size=(4,3,17))*10
    assert np.allclose(stratified_variance(raw,y),stratified_variance(raw-gamma[y],y))
