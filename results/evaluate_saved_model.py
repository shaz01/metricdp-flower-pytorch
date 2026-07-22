#!/usr/bin/env python3
"""Evaluate a transient final PaperCNN, save raw predictions + rich metrics atomically."""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
import numpy as np
import torch

from experiments.reproduce.dataset.alzheimer import AlzheimerDataModule, CLASS_NAMES
from experiments.reproduce.paper_cnn import PaperCNN


def predict(model, loader, device):
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            probs = model(images).cpu().numpy()
            ys.append(labels.numpy())
            ps.append(probs)
    return np.concatenate(ys).astype(np.int64), np.concatenate(ps).astype(np.float32)


def roc_curve_binary(y, score):
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    positives = int(y.sum()); negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    y = y[order]; score = score[order]
    distinct = np.where(np.diff(score))[0]
    idx = np.r_[distinct, len(score)-1]
    tps = np.cumsum(y)[idx].astype(float)
    fps = (1 + idx - tps).astype(float)
    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    # JSON has no representation for +/- infinity; null marks the two
    # synthetic endpoint thresholds while all observed-score thresholds remain.
    thresholds = [None, *score[idx].tolist(), None]
    auc = float(np.trapezoid(tpr, fpr))
    return {"auc": auc, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds}


def metrics(y, probabilities):
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    pred = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    support = cm.sum(axis=1)
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=(precision+recall) != 0)
    total = int(len(y)); weights = support / max(total, 1)
    clipped = np.clip(probabilities[np.arange(total), y], np.finfo(np.float64).tiny, 1.0)
    per_class_roc = {}
    aucs, auc_weights = [], []
    for c in range(n_classes):
        roc = roc_curve_binary((y == c).astype(np.int8), probabilities[:, c])
        per_class_roc[str(c)] = roc
        if roc is not None:
            aucs.append(roc["auc"]); auc_weights.append(float(weights[c]))
    micro_roc = roc_curve_binary(np.eye(n_classes, dtype=np.int8)[y].ravel(), probabilities.ravel())
    weighted_auc = None
    if aucs:
        denom = sum(auc_weights)
        weighted_auc = float(sum(a*w for a,w in zip(aucs,auc_weights)) / denom) if denom else None
    return {
        "num_examples": total,
        "accuracy": float((pred == y).mean()),
        "log_loss": float(-np.log(clipped).mean()),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            str(c): {"name": CLASS_NAMES[c], "support": int(support[c]), "precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c])}
            for c in range(n_classes)
        },
        "averages": {
            "macro": {"precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean())},
            "weighted": {"precision": float(np.dot(precision,weights)), "recall": float(np.dot(recall,weights)), "f1": float(np.dot(f1,weights))},
            "micro": {"precision": float(tp.sum()/max(total,1)), "recall": float(tp.sum()/max(total,1)), "f1": float(tp.sum()/max(total,1))},
        },
        "roc_ovr": {"macro_auc": float(np.mean(aucs)) if aucs else None, "weighted_auc": weighted_auc, "micro": micro_roc, "per_class": per_class_roc},
    }


def atomic_json(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_npz(path, arrays):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--run-json',required=True); ap.add_argument('--evaluation-json',required=True); ap.add_argument('--predictions',required=True)
    args=ap.parse_args()
    Path(args.evaluation_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
    run=json.loads(Path(args.run_json).read_text()); meta=run['metadata']; seed=int(meta['seed']); partition=meta['partition_mode']
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state=torch.load(args.model,map_location='cpu',weights_only=True)
    model=PaperCNN(); model.load_state_dict(state); model.to(device)
    dm=AlzheimerDataModule()
    _, server_loader=dm.server_loaders(batch_size=32,seed=seed,max_samples=0)
    sy,sp=predict(model,server_loader,device)
    arrays={'server_y_true':sy,'server_probabilities':sp,'server_y_pred':sp.argmax(1).astype(np.int64)}
    clients=[]; all_y=[]; all_p=[]
    for client_id in range(4):
        _, loader=dm.client_loaders(client_id,num_partitions=4,partition_mode=partition,batch_size=32,seed=seed,partition_profile='auto',client_weights=None,max_samples=0)
        y,p=predict(model,loader,device); all_y.append(y); all_p.append(p)
        arrays[f'client_{client_id}_y_true']=y; arrays[f'client_{client_id}_probabilities']=p; arrays[f'client_{client_id}_y_pred']=p.argmax(1).astype(np.int64)
        clients.append({'client_id':client_id, **metrics(y,p)})
    cy=np.concatenate(all_y); cp=np.concatenate(all_p)
    arrays['clients_combined_y_true']=cy; arrays['clients_combined_probabilities']=cp; arrays['clients_combined_y_pred']=cp.argmax(1).astype(np.int64)
    output={'schema_version':1,'run_name':meta['run_name'],'source_run_json':str(Path(args.run_json).resolve()),'class_names':list(CLASS_NAMES),'server_final_test':metrics(sy,sp),'clients_combined_test':metrics(cy,cp),'clients':clients,'raw_predictions':str(Path(args.predictions).resolve())}
    history = run['server_evaluate_metrics']
    final_round = max(int(key) for key in history if int(key) > 0)
    recorded_accuracy = float(history[str(final_round)]['accuracy'])
    if not math.isclose(output['server_final_test']['accuracy'], recorded_accuracy, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Postprocessed accuracy {output['server_final_test']['accuracy']} != recorded accuracy {recorded_accuracy}")
    output['validated_against_run_json'] = {'round': final_round, 'accuracy': recorded_accuracy}
    atomic_npz(args.predictions,arrays); atomic_json(args.evaluation_json,output)
    # Validate both artifacts before caller removes the transient model.
    json.loads(Path(args.evaluation_json).read_text())
    with np.load(args.predictions) as check:
        assert len(check['server_y_true']) == output['server_final_test']['num_examples']
    print(json.dumps({'server_accuracy':output['server_final_test']['accuracy'],'server_macro_f1':output['server_final_test']['averages']['macro']['f1'],'clients_accuracy':output['clients_combined_test']['accuracy']}))
if __name__=='__main__': main()
