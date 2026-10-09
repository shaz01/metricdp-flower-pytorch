import numpy as np
from scipy.special import ndtr
from research.calculations.public_residual_probe import (
    public_hessian, public_geometry, local_residual, projected_query,
    query_offset, bounded_public, public_scale,
)
from research.calculations.client_energy_filter_probe import C, scores


def test_empty_finetuning_preserves_reference_and_subspace():
    rng=np.random.default_rng(8);b=rng.normal(size=(3,17))*.1
    x=rng.normal(size=(40,17));y=np.tile(np.arange(4),10)
    h=public_hessian(x,b);geometry=public_geometry(h,3,'fisher')
    assert np.array_equal(local_residual(x[:0],y[:0],b,geometry['basis'],20,True),np.zeros((3,17)))
    delta=local_residual(x,y,b,geometry['basis'],20,True)
    projector=geometry['basis']@geometry['basis'].T
    assert np.allclose(delta.ravel(),projector@delta.ravel(),atol=1e-12)


def test_centered_absolute_in_identity_and_out_baseline_imputation():
    rng=np.random.default_rng(21);b=rng.normal(size=(3,17));teachers=rng.normal(size=(8,3,17))
    geometry=public_geometry(np.eye(51),12,'euclidean');gain=.3
    absolute=projected_query(teachers,b,geometry,'absolute')
    centered=projected_query(teachers,b,geometry,'centered')
    decode=lambda q: query_offset(b,geometry,'absolute',gain)+gain*(q.mean(axis=0)@geometry['decoder']).reshape(3,17)
    decoded_center=b+gain*(centered.mean(axis=0)@geometry['decoder']).reshape(3,17)
    assert np.allclose(decode(absolute),decoded_center)
    aout=absolute.copy();aout[0]=0;cout=centered.copy();cout[0]=0
    projected_b=(b.ravel()@geometry['encoder']@geometry['decoder']).reshape(3,17)
    assert np.allclose(b+gain*(cout.mean(axis=0)@geometry['decoder']).reshape(3,17)-decode(aout),gain*projected_b/8)


def test_public_hessian_matches_loss_directional_second_derivative():
    rng=np.random.default_rng(9);x=rng.normal(size=(40,17));y=np.tile(np.arange(4),10)
    b=rng.normal(size=(3,17))*.1;v=rng.normal(size=(3,17));step=1e-4
    h=public_hessian(x,b)
    second=(scores(x,y,(b+step*v)[None])[0][0]-2*scores(x,y,b[None])[0][0]+scores(x,y,(b-step*v)[None])[0][0])/step**2
    assert np.allclose(second,v.ravel()@h@v.ravel(),rtol=1e-6,atol=1e-6)
    assert np.allclose(h,h.T) and np.linalg.eigvalsh(h).min()>-1e-12


def test_whole_client_public_cap_and_gaussian_ceiling():
    rng=np.random.default_rng(17);query=rng.normal(size=(8,12));query[0]=0
    for cap in (.025,.4,1.6):
        signal,factors=bounded_public(query,cap)
        assert np.all(np.linalg.norm(signal,axis=1)<=cap/8+1e-12)
        assert np.array_equal(signal[0],np.zeros(12))
        sigma=public_scale(cap,.55)
        auc=ndtr(np.linalg.norm(signal,axis=1)/(np.sqrt(2)*sigma))
        assert auc.max()<=.55+1e-12 and auc[0]==.5


def test_public_geometry_encoder_decoder_and_fisher_metric():
    rng=np.random.default_rng(92);m=rng.normal(size=(51,51));h=m.T@m
    for dimension in (1,3,12,51):
        g=public_geometry(h,dimension,'fisher')
        projector=g['basis']@g['basis'].T
        assert np.allclose(g['encoder']@g['decoder'],projector,atol=1e-12)
        assert np.allclose(g['encoder'].T@g['encoder'],np.diag(g['eigenvalues']+.01))


def test_empty_mask_zeros_centered_query_instead_of_negative_reference():
    rng=np.random.default_rng(25);b=rng.normal(size=(3,17))
    teachers=np.zeros((8,3,17));g=public_geometry(np.eye(51),12,'euclidean')
    q=projected_query(teachers,b,g,'centered',active_mask=[False]+[True]*7)
    assert np.array_equal(q[0],np.zeros(12))
    assert np.linalg.norm(q[1])>0
