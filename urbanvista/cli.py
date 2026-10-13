from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from .utils import load_config,save_json
from .models import ALL_MODELS


def create_smoke_data(destination,patch_size=32,observations=2,n=30):
    """EXPLICITLY SYNTHETIC smoke fixtures, never paper results or real-data benchmark."""
    from .data import FEATURE_NAMES,grouped_split,compute_feature_stats
    root=Path(destination)
    (root/'patches').mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(909)
    groups=[]
    for i in range(n):
        lab=i%5
        images=rng.random((observations,4,patch_size,patch_size),dtype=np.float32)*.3
        images[:,lab%4] += .4
        mask=np.full((patch_size,patch_size),lab,dtype=np.uint8)
        mask[:patch_size//4,:patch_size//4]=(lab+1)%5
        feat=rng.normal(size=(len(FEATURE_NAMES),)).astype(np.float32)
        np.savez_compressed(root/'patches'/f'patch_{i:04d}.npz',images=images,mask=mask,
                            label=np.uint8(lab),osm=feat)
        groups.append(f'{i//5}')
    split=grouped_split(groups,seed=42)
    save_json(root/'split.json',split)
    save_json(root/'feature_stats.json',compute_feature_stats(root,split))
    save_json(root/'metadata.json',{'dataset_type':'SYNTHETIC_TEST_ONLY',
              'classes':['test_a','test_b','test_c','test_d','test_e'],'patch_size':patch_size,
              'n_dates':observations,'bands':['B02','B03','B04','B08']})
    return root


def main(argv=None):
    parser=argparse.ArgumentParser(description='UrbanVista end-to-end reproducibility CLI')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Download REAL Sentinel-2/OSM data and WorldCover or user label masks')
    p.add_argument('--config',default='configs/berlin_worldcover_2021.yaml')
    p.add_argument('--labels',default=None,help='Independent georeferenced class_id=0..4 polygons, GeoJSON or GPKG')
    r=sub.add_parser('run',help='Train baselines/ablation variants and produce genuine held-out metrics')
    r.add_argument('--config',default='configs/berlin_worldcover_2021.yaml')
    r.add_argument('--models',nargs='+',default=['urbanvista_full'],
                   help='Model names, "ablations" or "all"')
    r.add_argument('--seeds',nargs='+',type=int,default=None)
    r.add_argument('--runs',default='runs')
    q=sub.add_parser('predict',help='Predict an aligned prepared patch from a saved checkpoint')
    q.add_argument('--checkpoint',required=True)
    q.add_argument('--patch',required=True)
    q.add_argument('--dataset-root',required=True)
    q.add_argument('--output',default='predictions')
    q.add_argument('--device',default='auto')
    s=sub.add_parser('smoke-test',help='Run a real training loop on explicitly synthetic test data')
    s.add_argument('--workdir',default=None)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        from .geospatial import prepare
        prepare(load_config(args.config),labels_file=args.labels)
    elif args.command=='run':
        from .experiment import run_suite
        models=args.models
        if models==['ablations']: models=list(ALL_MODELS[:5])
        elif models==['all']: models=ALL_MODELS
        unknown=set(models)-set(ALL_MODELS)
        if unknown:parser.error('Unknown models: '+str(sorted(unknown)))
        run_suite(load_config(args.config),models,seeds=args.seeds,run_base=args.runs)
    elif args.command=='predict':
        from .inference import predict
        target=predict(args.checkpoint,args.patch,args.output,args.dataset_root,device=args.device)
        print('Prediction saved in',target)
    elif args.command=='smoke-test':
        from .experiment import train_one
        import tempfile
        temporary=None
        if args.workdir is None:
            temporary=tempfile.TemporaryDirectory(prefix='urbanvista_smoke_')
            directory=Path(temporary.name)
        else:
            directory=Path(args.workdir)
        root=create_smoke_data(directory/'synthetic')
        cfg={'name':'SMOKE_TEST_NOT_PAPER_DATA','output_dir':str(root),'bands':['B02','B03','B04','B08'],
             'patch_size':32,'training':{'epochs':1,'batch_size':4,'learning_rate':1e-4,'weight_decay':0,
                 'patience':1,'num_workers':0,'classification_weight':1.,'segmentation_weight':1.,
                 'seeds':[11],'device':'cpu'}}
        result=train_one(cfg,'urbanvista_full',11,run_base=directory/'runs')
        print('Synthetic smoke test passed. Classification accuracy: '+str(result['accuracy']))
        print('This synthetic metric is NOT an experimental result for the UrbanVista paper.')
        if temporary: temporary.cleanup()

if __name__=='__main__': main()
