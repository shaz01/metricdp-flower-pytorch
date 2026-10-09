import numpy as np
from scipy.special import ndtr
from research.calculations.class_conditional_probe import weighted_scores, client_queries, public_offset, transform, prior_correct
from research.calculations.public_residual_probe import public_geometry, bounded_public, public_scale
from research.calculations.client_energy_filter_probe import C, scores


def test_weighted_objective():
    rng=np.random.default_rng(4); x=rng.normal(size=(12,17)); y=np.repeat(np.arange(4),3); b=rng.normal(size=(2,3,17))
    p=np.array([.4,.2,.2,.2]); got=weighted_scores(x,y,b,p)
    expect=np.array([scores(x[y==k],y[y==k],b) for k in range(4)])
    assert np.allclose(got[0],p@expect[:,0]); assert np.allclose(got[1],p@expect[:,1])


def test_centering_and_empty_contract():
    rng=np.random.default_rng(5); x=rng.normal(size=(24,17)); y=np.tile(np.arange(4),6); b=rng.normal(size=(3,17)); gamma=rng.normal(size=(4,3,17)); p=np.ones(4)/4
    fits=[np.arange(24)]*8; q,priors=client_queries(x,y,fits,b,gamma,p)
    g=np.einsum('k,kad->ad',p,gamma)
    assert np.allclose(q['raw'].mean(0),q['class_center'].mean(0)+g)
    assert np.allclose(q['target_balanced']-g,q['target_center'])
    fits[-1]=np.array([],dtype=int); q,priors=client_queries(x,y,fits,b,gamma,p)
    assert all(np.count_nonzero(a[-1])==0 for a in q.values())
    assert np.allclose(q['class_center'].mean(0)+g-q['raw'].mean(0),g/8)


def test_projection_and_public_cap():
    h=np.diag(np.arange(1,52)); gamma=np.ones((3,17)); geometry=public_geometry(h,3,'fisher')
    offset=public_offset('raw',gamma,geometry)
    assert np.allclose(offset.ravel()+gamma.ravel()@geometry['encoder']@geometry['decoder'],gamma.ravel())
    rng=np.random.default_rng(1); signal,_=bounded_public(rng.normal(size=(8,3)),.003); sigma=public_scale(.003,.55)
    assert np.all(ndtr(np.linalg.norm(signal,axis=1)/(np.sqrt(2)*sigma))<=.55+1e-12)


def test_public_stress_and_bayes_control():
    x=np.ones((3,17)); out=transform(x,True); assert (out[:,[5,6,9,10]]==0).all(); assert (out[:,-1]==1).all(); assert (x==1).all()
    p=np.array([.4,.2,.2,.2]); b=np.zeros((3,17)); corrected=prior_correct(b,p); logits=C@corrected[:,-1]
    prob=np.exp(logits); prob/=prob.sum(); assert np.allclose(prob,p)


def test_skew_missing_classes_and_true_subset():
    from research.calculations.class_conditional_probe import scenario_fits
    from research.calculations.public_residual_signal_diagnostic import example_gradients
    rng=np.random.default_rng(11); x=rng.normal(size=(16,17)); y=np.array([0]*8+[1]*4+[2]*4); b=rng.normal(size=(3,17))*.1; gamma=rng.normal(size=(4,3,17)); p=np.array([.4,.2,.2,.2])
    q,prior=client_queries(x,y,[np.arange(16)]*8,b,gamma,p)
    grad=example_gradients(x,y,b)
    expected=sum(p[k]*(grad[y==k].mean(0) if np.any(y==k) else gamma[k]) for k in range(4))
    assert np.allclose(q['target_balanced'][0],expected)
    assert np.allclose(q['class_center'][0],grad.mean(0)-np.einsum('k,kad->ad',prior[0],gamma))
    assert not np.allclose(q['class_center'],q['fixed_center'])
    fit=scenario_fits([np.arange(16)],y,'prior')[0]
    assert len(fit)==12 and len(set(fit))==12 and set(fit)<=set(range(16))


def test_gradient_matches_directional_derivative():
    from research.calculations.public_residual_signal_diagnostic import example_gradients
    rng=np.random.default_rng(15); x=rng.normal(size=(16,17)); y=np.tile(np.arange(4),4); b=rng.normal(size=(3,17))*.1; v=rng.normal(size=b.shape); step=1e-5
    derivative=(scores(x,y,(b+step*v)[None])[0][0]-scores(x,y,(b-step*v)[None])[0][0])/(2*step)
    assert np.allclose(derivative,np.sum(example_gradients(x,y,b).mean(0)*v),atol=1e-8)
