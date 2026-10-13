import json
import numpy as np
import pytest
import torch
from shapely.geometry import box
from urbanvista.data import grouped_split, PatchDataset
from urbanvista.geospatial import bearing_axial, tile_names, osm_patch_features
from urbanvista.metrics import conf_update, cls_scores, seg_scores, dice_loss
from urbanvista.models import create_model, VARIANTS
from urbanvista.cli import create_smoke_data


def test_geographic_block_split():
    groups=[f'block{i//5}' for i in range(75)]
    split=grouped_split(groups,42)
    sets=[set(groups[i] for i in split[key]) for key in ('train','val','test')]
    assert all(sets[i].isdisjoint(sets[j]) for i,j in [(0,1),(0,2),(1,2)])
    assert sum(len(s) for s in split.values())==75


def test_orientation_symmetry_and_tiles():
    from math import pi, cos
    north=bearing_axial(box(0,0,4,20));east=bearing_axial(box(0,0,20,4))
    assert north is not None and east is not None
    assert np.isclose(cos(2*north),1)
    assert np.isclose(cos(2*east),-1)
    assert bearing_axial(box(0,0,10,10)) is None
    assert tile_names([13.25,52.45,13.55,52.59])==['N51E012']


def test_models_forward_shapes():
    torch.set_num_threads(2)
    x=torch.rand(2,2,4,32,32)
    osm=torch.randn(2,10)
    for name in [*VARIANTS,'unet','segnet','deeplabv3plus']:
        model=create_model(name,channels=4,img_size=32)
        with torch.no_grad(): c,s=model(x,osm)
        assert c.shape==(2,5),name
        assert s is not None and s.shape==(2,5,32,32),name


def test_confusion_metrics_and_ignore():
    m=np.zeros((5,5),dtype=np.int64)
    conf_update(m,np.array([0,1,2,255]),np.array([0,1,0,1]))
    assert m.sum()==3
    assert cls_scores(m)['accuracy']==pytest.approx(2/3)
    assert 0<=seg_scores(m)['mean_iou']<=1
    target=torch.full((1,8,8),255,dtype=torch.long)
    assert torch.isfinite(dice_loss(torch.randn(1,5,8,8),target))


def test_smoke_patches(tmp_path):
    root=create_smoke_data(tmp_path/'synthetic',32,2,30)
    training=PatchDataset(root,'train')
    sample=training[0]
    assert sample['images'].shape==(2,4,32,32)
    assert sample['osm'].shape==(10,)
    meta=json.loads((root/'metadata.json').read_text())
    assert meta['dataset_type']=='SYNTHETIC_TEST_ONLY'
