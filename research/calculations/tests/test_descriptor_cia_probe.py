import numpy as np
from research.calculations.descriptor_cia_probe import auc_statistics, gaussian_llr, peer_residual


def test_auc_is_all_pairs_with_ties_not_paired_concordance():
    # Paired concordance is 1; pooled ROC-AUC is .75.
    assert auc_statistics(np.array([2.,4.]),np.array([1.,3.]))['auc']==.75
    assert auc_statistics(np.ones(8),np.ones(7))['auc']==.5
    assert auc_statistics(np.zeros(8),np.ones(7))['auc']==0


def test_peer_removes_only_own_upload_and_dummy_keeps_noise():
    signal=np.arange(24.).reshape(8,3);noise=np.ones((4,8,3))
    objects=signal[None]+noise
    residual=peer_residual(objects,7)
    assert np.allclose(residual,(signal[:7]+1).sum(axis=0))
    objects[:,0]-=signal[0]
    assert np.allclose(peer_residual(objects,7),residual-signal[0])
    assert np.all(objects[:,0]==noise[:,0])


def test_gaussian_llr_accounts_for_variance_change():
    x=np.array([[0.,0.],[3.,0.]])
    score=gaussian_llr(x,np.zeros(2),np.zeros(2),1.,2.)
    # Equal means: wider IN distribution has lower density centrally,
    # higher density in tails; a mean-only attack would miss this.
    assert score[0]<0<score[1]
    score=gaussian_llr(np.array([[0.],[1.]]),np.array([0.]),np.array([1.]),1.,1.)
    assert np.allclose(score,[-.5,.5])


def test_frozen_targets_cover_all_label_stress_dominant_classes():
    from research.calculations.descriptor_cia_probe import TARGETS
    assert {target%4 for target in TARGETS}=={0,1,2,3}


def test_low_fpr_keeps_attainable_collinear_thresholds():
    # A pruned ROC curve drops attainable intermediate operating points.
    result=auc_statistics(np.arange(100.),np.arange(100.))
    assert result['tpr_at_fpr01']==.01
    assert result['tpr_at_fpr05']==.05
