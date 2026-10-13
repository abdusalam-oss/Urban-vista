"""Supervised training, held-out evaluation and seed-paired reporting.
All metrics are computed from model outputs; none are imported from the paper.
"""
from __future__ import annotations
import copy
import csv
import json
import platform
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import PatchDataset, WORLDCOVER_CLASSES
from .models import create_model, ALL_MODELS
from .metrics import conf_update,cls_scores,seg_scores,dice_loss,paired_test
from .plots import confusion_figure,segmentation_example,learning_curves
from .utils import get_device,save_json,seed_all


def loader(root, split, batch_size, workers=0):
    return DataLoader(PatchDataset(root,split,augment=(split=='train')),
                      batch_size=batch_size,shuffle=(split=='train'),num_workers=workers,
                      pin_memory=torch.cuda.is_available())


def epoch_pass(model, batches, device, optimizer=None, class_weight=1.0, mask_weight=1.0):
    train=optimizer is not None
    model.train(train)
    n=0; total=0.; class_cm=np.zeros((5,5),dtype=np.int64)
    mask_cm=np.zeros((5,5),dtype=np.int64)
    with torch.set_grad_enabled(train):
        for batch in batches:
            images=batch['images'].to(device); osm=batch['osm'].to(device)
            target=batch['label'].to(device); mask=batch['mask'].to(device)
            if train: optimizer.zero_grad(set_to_none=True)
            cls,seg=model(images,osm)
            loss=class_weight*nn.functional.cross_entropy(cls,target)
            if seg is not None and mask_weight>0:
                ce=nn.functional.cross_entropy(seg,mask,ignore_index=255)
                loss=loss+mask_weight*(ce+dice_loss(seg,mask))/2
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
            b=len(target);n+=b;total+=float(loss.detach().item())*b
            conf_update(class_cm,target.detach().cpu().numpy(),cls.argmax(1).detach().cpu().numpy())
            if seg is not None:
                conf_update(mask_cm,mask.detach().cpu().numpy(),seg.argmax(1).detach().cpu().numpy())
    result={'loss':total/n if n else None,**cls_scores(class_cm),'class_confusion':class_cm.tolist()}
    if mask_cm.sum():
        result.update(seg_scores(mask_cm));result['pixel_confusion']=mask_cm.tolist()
    else:
        result.update({'mean_iou':None,'per_class_iou':None,'pixel_accuracy':None,'pixel_confusion':None})
    return result


def train_one(cfg, model_name, seed, run_base='runs'):
    if model_name not in ALL_MODELS: raise ValueError(f'Unknown model {model_name}')
    seed_all(int(seed))
    root=Path(cfg['output_dir'])
    if not (root/'metadata.json').exists():
        raise FileNotFoundError(f'{root}/metadata.json missing. Run urbanvista prepare first.')
    meta=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    params=cfg['training']; device=get_device(params.get('device','auto'))
    output=Path(run_base)/cfg['name']/model_name/f'seed_{seed}'
    output.mkdir(parents=True,exist_ok=True)
    trainloader=loader(root,'train',int(params['batch_size']),int(params.get('num_workers',0)))
    valloader=loader(root,'val',int(params['batch_size']),int(params.get('num_workers',0)))
    testloader=loader(root,'test',int(params['batch_size']),int(params.get('num_workers',0)))
    model=create_model(model_name,channels=len(cfg['bands']),img_size=int(cfg['patch_size'])).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(params['learning_rate']),
                               weight_decay=float(params.get('weight_decay',0)))
    history=[];best=float('inf');best_state=None;wait=0
    start=time.time()
    for epoch in range(1,int(params['epochs'])+1):
        tr=epoch_pass(model,trainloader,device,optimizer,
                      float(params.get('classification_weight',1.)),float(params.get('segmentation_weight',1.)))
        val=epoch_pass(model,valloader,device,None,
                       float(params.get('classification_weight',1.)),float(params.get('segmentation_weight',1.)))
        history.append({'epoch':epoch,'train_loss':tr['loss'],'val_loss':val['loss'],
                        'val_accuracy':val['accuracy'],'val_f1_macro':val['f1_macro'],'val_mean_iou':val['mean_iou']})
        print(f'[{model_name} seed={seed}] epoch {epoch}: train={tr["loss"]:.4f} val={val["loss"]:.4f}',flush=True)
        if val['loss'] < best - 1e-5:
            best=val['loss'];wait=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=int(params['patience']):
                print('Early stopping',flush=True);break
    if best_state is None: raise RuntimeError('No model checkpoint obtained')
    model.load_state_dict(best_state)
    result=epoch_pass(model,testloader,device,None,
                      float(params.get('classification_weight',1.)),float(params.get('segmentation_weight',1.)))
    duration=time.time()-start
    result.update({'model':model_name,'seed':int(seed),'epochs':len(history),
                   'seconds_train_and_test':round(duration,3),'test_samples':len(testloader.dataset),
                   'dataset_type':meta['dataset_type'],'checkpoint':str(output/'model.pt')})
    torch.save({'state_dict':best_state,'model_name':model_name,'seed':int(seed),
                'n_bands':len(cfg['bands']),'patch_size':int(cfg['patch_size']),
                'class_names':meta['classes']},output/'model.pt')
    save_json(output/'test_metrics.json',result)
    save_json(output/'learning_history.json',history)
    save_json(output/'run_config.json',cfg)
    save_json(output/'environment.json',{'python':platform.python_version(),'pytorch':torch.__version__,
                    'cuda_available':torch.cuda.is_available(),'device':str(device)})
    class_names=meta['classes']
    confusion_figure(result['class_confusion'],class_names,output/'classification_confusion.png')
    learning_curves(history,output/'training.png')
    if result['pixel_confusion'] is not None:
        confusion_figure(result['pixel_confusion'],class_names,output/'segmentation_confusion.png')
        # Genuine held-out sample. Figure is NOT a diagram or generative illustration.
        sample=next(iter(testloader))
        model.eval()
        with torch.no_grad():
            _,seg=model(sample['images'].to(device),sample['osm'].to(device))
        if seg is not None:
            rgb=sample['images'][0,-1,[2,1,0]].permute(1,2,0).numpy()
            segmentation_example(rgb,sample['mask'][0].numpy(),seg[0].argmax(0).cpu().numpy(),
                                 class_names,output/'segmentation_example.png')
    return result


def run_suite(cfg, model_names, seeds=None, run_base='runs'):
    if seeds is None: seeds=cfg['training']['seeds']
    results=[]
    for name in model_names:
        for seed in seeds:
            results.append(train_one(cfg,name,int(seed),run_base=run_base))
            save_summary(results,Path(run_base)/cfg['name'])
    return results


def save_summary(results, base):
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    simple=['model','seed','accuracy','precision_macro','recall_macro','f1_macro','mean_iou',
            'pixel_accuracy','epochs','seconds_train_and_test','test_samples','dataset_type']
    with open(base/'results_by_seed.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=simple);writer.writeheader()
        for r in results: writer.writerow({key:r.get(key) for key in simple})
    comparisons=[]; summaries=[]
    models=sorted(set(r['model'] for r in results))
    for name in models:
        rows=[r for r in results if r['model']==name]
        rec={'model':name,'n_runs':len(rows)}
        for metric in ['accuracy','f1_macro','mean_iou']:
            nums=np.array([r[metric] for r in rows if r[metric] is not None])
            rec[metric+'_mean']=float(nums.mean()) if len(nums) else None
            rec[metric+'_sd']=float(nums.std(ddof=1)) if len(nums)>1 else None
        summaries.append(rec)
    # compare full UrbanVista vs each baseline/ablation on matched random seeds
    main={r['seed']:r for r in results if r['model']=='urbanvista_full'}
    for name in models:
        if name=='urbanvista_full':continue
        other={r['seed']:r for r in results if r['model']==name}
        matched=sorted(set(main)&set(other))
        if len(matched)>=2:
            comparisons.append({'baseline':name,'paired_seeds':matched,
               'macro_f1':paired_test([main[s]['f1_macro'] for s in matched],[other[s]['f1_macro'] for s in matched])})
    save_json(base/'aggregate_summary.json',{'models':summaries,'paired_tests':comparisons,
        'interpretation':'No p-value establishes significance unless computed from real executed independent runs. Five seeds yield minimum exact two-sided Wilcoxon p=0.0625 without ties.'})
