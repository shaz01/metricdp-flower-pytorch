import numpy as np
from scipy.special import ndtr,ndtri
from research.calculations.matched_cia_probe import gaussian_std, radial_scale_ratio, noise_objects


def test_gaussian_risk_mapping_and_validation():
    for risk in (.55,.65,.8):
        std=gaussian_std(.2,risk)
        assert abs(ndtr(.2/(np.sqrt(2)*std))-risk)<1e-12
        assert gaussian_std(.4,risk)==2*std


def test_radial_uses_whole_vector_and_variance_control():
    signal=np.zeros((8,12));signal[0,0]=.1
    s=.03
    radial=noise_objects(signal,'radial',s,12000,77)
    gaussian=noise_objects(signal,'gaussian',np.sqrt(13)*s,12000,77)
    expected=8/7*13*s*s
    assert abs(np.var(radial.sum(axis=1),axis=0).mean()/expected-1)<.04
    assert abs(np.var(gaussian.sum(axis=1),axis=0).mean()/expected-1)<.04
    assert np.allclose(noise_objects(2*signal,'radial',2*s,10,77),2*noise_objects(signal,'radial',s,10,77))


def test_radial_calibration_is_seed_fixed_and_less_noise_for_higher_risk():
    assert radial_scale_ratio(12,.55)>radial_scale_ratio(12,.8)>0
    assert radial_scale_ratio(12,.65)==radial_scale_ratio(12,.65)



def test_efficient_statistics_equal_independent_all_pairs_statistic():
    from research.calculations.matched_cia_probe import auc_statistics
    from research.calculations.descriptor_cia_probe import auc_statistics as direct
    rng=np.random.default_rng(22);p=rng.integers(0,7,123);n=rng.integers(0,7,117)
    actual=auc_statistics(p,n);expected=direct(p,n)
    for key in expected:assert np.allclose(actual[key],expected[key],rtol=0,atol=1e-15)


def test_dummy_retains_noise_and_peer_residual_has_full_floor():
    signal=np.zeros((8,12));signal[0,0]=.1;s=.03
    objects=noise_objects(signal,'radial',s,12000,53)
    out=signal.copy();out[0]=0
    empty=noise_objects(out,'radial',s,12000,53)
    assert np.allclose(objects-empty,signal[None]-out[None])
    assert np.any(empty[:,0]!=0)
    residual=objects.sum(axis=1)-objects[:,7]
    assert abs(residual.var(axis=0).mean()/(13*s*s)-1)<.04
    central=noise_objects(signal,'radial',s,12000,53,central=True).sum(axis=1)
    assert abs(central.var(axis=0).mean()/(13*s*s)-1)<.04
