"""Dataset-wide, confusion-matrix aggregated classification and segmentation scores."""
import numpy as np
import torch
from scipy.stats import wilcoxon


def conf_update(matrix, gt, pred, classes=5, ignore=255):
    gt = np.asarray(gt).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    keep = (gt != ignore) & (gt >= 0) & (gt < classes) & (pred >= 0) & (pred < classes)
    matrix += np.bincount(classes*gt[keep].astype(np.int64)+pred[keep].astype(np.int64),
                          minlength=classes*classes).reshape(classes,classes)
    return matrix


def cls_scores(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    tp=np.diag(m); row=m.sum(1); col=m.sum(0)
    with np.errstate(invalid='ignore',divide='ignore'):
        prec=np.divide(tp,col,out=np.zeros_like(tp),where=col>0)
        rec=np.divide(tp,row,out=np.zeros_like(tp),where=row>0)
        f1=np.divide(2*prec*rec,prec+rec,out=np.zeros_like(tp),where=(prec+rec)>0)
    supported=row>0
    def avg(a): return float(a[supported].mean()) if supported.any() else 0.0
    return {'accuracy':float(tp.sum()/m.sum()) if m.sum() else 0.,
            'precision_macro':avg(prec), 'recall_macro':avg(rec), 'f1_macro':avg(f1),
            'class_support':row.astype(int).tolist(),
            'per_class_f1':f1.tolist()}


def seg_scores(matrix):
    m=np.asarray(matrix,dtype=np.float64)
    tp=np.diag(m); union=m.sum(0)+m.sum(1)-tp
    iou=np.divide(tp,union,out=np.zeros_like(tp),where=union>0)
    present=union>0
    return {'mean_iou':float(iou[present].mean()) if present.any() else 0.,
            'per_class_iou':iou.tolist(), 'pixel_accuracy':float(tp.sum()/m.sum()) if m.sum() else 0.}


def dice_loss(logits, target, classes=5, ignore=255):
    """Masked multi-class soft Dice over present classes only."""
    valid=(target != ignore)
    tgt=target.clamp(0,classes-1)
    probs=torch.softmax(logits,1)*valid.unsqueeze(1)
    hot=torch.nn.functional.one_hot(tgt, classes).permute(0,3,1,2).to(probs.dtype)
    hot=hot*valid.unsqueeze(1)
    dims=(0,2,3)
    intersect=(probs*hot).sum(dims)
    denom=probs.sum(dims)+hot.sum(dims)
    existing=hot.sum(dims)>0
    if not bool(existing.any()): return logits.sum()*0
    dice=(2*intersect+1)/(denom+1)
    return (1-dice[existing]).mean()


def paired_test(xs,ys):
    if len(xs) != len(ys) or len(xs)<2:
        return {'p_value':None,'reason':'Need at least 2 paired runs'}
    delta=np.array(xs)-np.array(ys)
    if np.allclose(delta,0): return {'p_value':1.,'reason':'all paired differences zero'}
    res=wilcoxon(xs,ys,alternative='two-sided')
    return {'p_value':float(res.pvalue),'n_pairs':len(xs),'test':'two-sided paired Wilcoxon'}
