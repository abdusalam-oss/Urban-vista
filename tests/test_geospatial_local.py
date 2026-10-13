"""Offline integration tests. Local fixtures are not research data."""
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import box, LineString
from urbanvista import geospatial


def test_reprojection_reader_local(tmp_path):
    local=tmp_path/'sample.tif'
    transform=from_origin(500000,6000000,10,10)
    with rasterio.open(local,'w',driver='GTiff',count=1,width=16,height=16,
                       crs='EPSG:32633',transform=transform,dtype='uint8') as dst:
        dst.write(np.full((16,16),32,dtype=np.uint8),1)
    arr=geospatial.read_asset(str(local),'EPSG:32633',transform,16,16,
                              Resampling.nearest,dtype=np.uint8)
    assert arr.shape==(16,16)
    assert np.all(arr==32)


def test_full_preparation_offline(monkeypatch,tmp_path):
    from urbanvista.utils import load_config
    config=load_config('configs/berlin_worldcover_2021.yaml')
    config.update({'output_dir':str(tmp_path/'prepared'),'patch_size':32,
                   'min_class_dominance':0.3,'max_patch_cloud_fraction':.5,
                   'geographic_block_patches':2,'n_dates':2})
    width=32*10;height=32*10
    tr=from_origin(500000,6000000,10,10)
    monkeypatch.setattr(geospatial,'projected_grid',lambda *a:('EPSG:32633',tr,width,height))
    rng=np.random.default_rng(101)
    im=rng.random((2,4,height,width),dtype=np.float32)
    valid=np.ones((2,height,width),dtype=bool)
    monkeypatch.setattr(geospatial,'acquire_sentinel',lambda *a:(im,valid,[{'id':'TEST_SCENE'}]*2))
    masks=np.empty((height,width),dtype=np.uint8)
    for r in range(10):
        for c in range(10):
            masks[r*32:(r+1)*32,c*32:(c+1)*32]=(r+c)%5
    monkeypatch.setattr(geospatial,'get_worldcover',lambda *a:masks)
    building=gpd.GeoDataFrame({'building':['yes'],'height':[10.]},geometry=[box(500100,5999900,500300,5999970)],crs='EPSG:32633')
    road=gpd.GeoDataFrame({'highway':['residential']},geometry=[LineString([(500000,5999910),(503200,5999910)])],crs='EPSG:32633')
    monkeypatch.setattr(geospatial,'fetch_osm',lambda *a:(building,road))
    out=geospatial.prepare(config)
    assert len(list((out/'patches').glob('*.npz')))==100
    metadata=json.loads((out/'metadata.json').read_text())
    assert metadata['n_patches']==100
    assert sum(metadata['split_sizes'].values())==100
    assert metadata['dataset_type']=='public_five_class_worldcover_land_cover'
