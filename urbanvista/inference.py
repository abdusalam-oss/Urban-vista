"""Inference for one already prepared georeferenced aligned patch (*.npz).
Does not invent unseen OSM features or geolocation for arbitrary images.
"""
import json
import re
from pathlib import Path
import numpy as np
import torch
from .models import create_model
from .utils import get_device, save_json


def predict(checkpoint, patch, output, dataset_root, device='auto'):
    ckpt_path=Path(checkpoint)
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    root=Path(dataset_root)
    if not (root/'feature_stats.json').is_file():
        raise FileNotFoundError('Real training-set feature statistics are required to normalize OSM inputs.')
    stats=json.loads((root/'feature_stats.json').read_text(encoding='utf-8'))
    npz=Path(patch)
    with np.load(npz) as data:
        images=data['images'].astype(np.float32)
        osm=data['osm'].astype(np.float32)
    if images.ndim!=4 or images.shape[1]!=ckpt['n_bands'] or images.shape[-1]!=ckpt['patch_size']:
        raise ValueError('Patch shape does not match checkpoint input bands or size')
    mu=np.asarray(stats['mean'],dtype=np.float32)
    sd=np.asarray(stats['std'],dtype=np.float32)
    osm=np.clip((osm-mu)/sd,-10,10)
    dev=get_device(device)
    model=create_model(ckpt['model_name'],channels=ckpt['n_bands'],
                       classes=len(ckpt['class_names']),img_size=ckpt['patch_size']).to(dev)
    model.load_state_dict(ckpt['state_dict'],strict=True)
    model.eval()
    with torch.inference_mode():
        cls,seg=model(torch.tensor(images[None],device=dev),torch.tensor(osm[None],device=dev))
        probs=torch.softmax(cls,1)[0].cpu().numpy()
        mask=seg[0].argmax(0).cpu().numpy().astype(np.uint8) if seg is not None else None
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    save_json(out/'prediction.json',{'patch':str(npz),'class_names':ckpt['class_names'],
          'prediction':ckpt['class_names'][int(probs.argmax())],
          'probabilities':{name:float(prob) for name,prob in zip(ckpt['class_names'],probs)},
          'segmentation_saved':mask is not None})
    if mask is not None:
        from PIL import Image
        Image.fromarray(mask).save(out/'segmentation_class_ids.png')
        metadata_file=root/'metadata.json'
        if metadata_file.exists():
            meta=json.loads(metadata_file.read_text(encoding='utf-8'))
            match=re.fullmatch(r'patch_(\d+)_(\d+)',npz.stem)
            if match and 'transform_gdal' in meta and 'crs' in meta:
                import rasterio
                from rasterio.windows import Window, transform as window_transform
                transform=rasterio.Affine.from_gdal(*meta['transform_gdal'])
                ps=int(ckpt['patch_size'])
                cell=Window(int(match.group(2))*ps,int(match.group(1))*ps,ps,ps)
                with rasterio.open(out/'segmentation_georeferenced.tif','w',driver='GTiff',
                                   width=ps,height=ps,count=1,dtype='uint8',crs=meta['crs'],
                                   transform=window_transform(cell,transform),
                                   nodata=255,compress='deflate') as dst:
                    dst.write(mask,1)
    return out
