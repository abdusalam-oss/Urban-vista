"""Acquire aligned real Sentinel-2, ESA WorldCover, and OSM features.

Default public benchmark predicts five WorldCover LAND-COVER types, not the five
urban LAND-USE types in the article. For the manuscript's task provide independent
GeoJSON/GPKG polygons labelled class_id=0..4 with --labels.
"""
from __future__ import annotations
from datetime import datetime, timezone
from itertools import islice
from math import atan2, ceil, cos, degrees, floor, log1p, pi, radians, sin, sqrt
from pathlib import Path
import json
import warnings
import numpy as np

from .data import FEATURE_NAMES, WORLDCOVER_MAP, WORLDCOVER_CLASSES, grouped_split, compute_feature_stats, IGNORE
from .utils import save_json


def _dependencies():
    try:
        import geopandas as gpd
        import osmnx as ox
        import rasterio
        from pystac_client import Client
        import planetary_computer as pc
    except ImportError as exc:
        raise RuntimeError('Missing geospatial extras. Run: pip install -e ".[all]"') from exc
    return gpd, ox, rasterio, Client, pc


def choose_scenes(items, n_dates, bbox):
    from shapely.geometry import shape, box
    roi = box(*bbox)
    eligible = [x for x in items if x.datetime is not None and x.geometry and
                shape(x.geometry).covers(roi) and all(k in x.assets for k in ('B02','B03','B04','B08','SCL'))]
    if len(eligible) < n_dates:
        raise RuntimeError(f'Only {len(eligible)} scenes cover the whole AOI, need {n_dates}. Shrink the bbox or widen the date window.')
    by_month = {}
    for item in eligible:
        month = item.datetime.strftime('%Y-%m')
        if month not in by_month or item.properties.get('eo:cloud_cover', 100) < by_month[month].properties.get('eo:cloud_cover', 100):
            by_month[month] = item
    monthly = sorted(by_month.values(), key=lambda x: x.datetime)
    if len(monthly) < n_dates:
        raise RuntimeError(f'Only {len(monthly)} distinct months cover the AOI; need {n_dates}.')
    positions = np.linspace(0, len(monthly)-1, n_dates).round().astype(int)
    return [monthly[int(i)] for i in positions]


def projected_grid(bbox, resolution_m, patch_size):
    import geopandas as gpd
    from shapely.geometry import Point
    from rasterio.warp import transform_bounds
    from rasterio.transform import from_origin
    west, south, east, north = bbox
    crs = gpd.GeoSeries([Point((west+east)/2, (south+north)/2)], crs='EPSG:4326').estimate_utm_crs()
    if crs is None:
        raise ValueError('Could not estimate a suitable UTM CRS')
    x0, y0, x1, y1 = transform_bounds('EPSG:4326', crs, *bbox, densify_pts=21)
    width = int((x1-x0)/resolution_m)//patch_size*patch_size
    height = int((y1-y0)/resolution_m)//patch_size*patch_size
    if width < patch_size * 3 or height < patch_size * 3:
        raise ValueError('AOI is too small for spatially separated train/validation/test sets')
    return crs, from_origin(x0, y1, resolution_m, resolution_m), width, height


def read_asset(href, crs, transform, width, height, resampling, *, fill=0, dtype=np.float32):
    import rasterio
    from rasterio.vrt import WarpedVRT
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_MAX_RETRY='3', GDAL_HTTP_RETRY_DELAY='2'):
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs=crs, transform=transform, width=width, height=height,
                           resampling=resampling, nodata=src.nodata, warp_extras={'NUM_THREADS':'2'}) as vrt:
                result = vrt.read(1, masked=True)
                # mask from source nodata is preserved and becomes fill
                return np.asarray(result.filled(fill), dtype=dtype)


def acquire_sentinel(cfg, crs, transform, width, height):
    _, _, _, Client, pc = _dependencies()
    from rasterio.enums import Resampling
    print('Searching real Sentinel-2 L2A COG assets (Microsoft Planetary Computer STAC)...', flush=True)
    catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
    search = catalog.search(collections=['sentinel-2-l2a'], bbox=cfg['bbox'],
                            datetime=f"{cfg['start_date']}/{cfg['end_date']}",
                            query={'eo:cloud_cover': {'lt': cfg['scene_cloud_max_pct']}}, max_items=500)
    scenes = choose_scenes(list(search.items()), cfg['n_dates'], cfg['bbox'])
    imgs = []
    valid = []
    scene_meta = []
    for item in scenes:
        print(f'  {item.id}: {item.datetime.date()} (cloud={item.properties.get("eo:cloud_cover")})', flush=True)
        arr = np.stack([read_asset(item.assets[b].href, crs, transform, width, height,
                                   Resampling.bilinear) for b in cfg['bands']], axis=0)
        scl = read_asset(item.assets['SCL'].href, crs, transform, width, height,
                         Resampling.nearest, dtype=np.uint8)
        ok = np.isin(scl, [4, 5, 6, 7]) & np.all(arr > 0, axis=0)
        arr = np.clip(arr / 10000., 0, 1)
        arr[:, ~ok] = 0
        imgs.append(arr.astype(np.float32)); valid.append(ok)
        scene_meta.append({'id': item.id, 'date': item.datetime.date().isoformat(),
                           'cloud_cover_scene_pct': item.properties.get('eo:cloud_cover'),
                           'assets': list(cfg['bands']) + ['SCL']})
    return np.stack(imgs), np.stack(valid), scene_meta


def tile_names(bbox):
    west, south, east, north = bbox
    lat_lows = range(floor(south/3)*3, floor((north-1e-8)/3)*3+1, 3)
    lon_lows = range(floor(west/3)*3, floor((east-1e-8)/3)*3+1, 3)
    return [f'{"N" if y>=0 else "S"}{abs(y):02d}{"E" if x>=0 else "W"}{abs(x):03d}'
            for y in lat_lows for x in lon_lows]


def get_worldcover(cfg, crs, transform, width, height):
    from rasterio.enums import Resampling
    raw = np.full((height,width), IGNORE, dtype=np.uint8)
    for tile in tile_names(cfg['bbox']):
        url = (f'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/'
               f'ESA_WorldCover_10m_2021_v200_{tile}_Map.tif')
        print(f'Reading WorldCover from {url}', flush=True)
        try:
            a = read_asset(url, crs, transform, width, height, Resampling.nearest,
                           fill=0, dtype=np.uint8)
        except Exception as e:
            raise RuntimeError(f'WorldCover tile {tile} could not be read: {e}') from e
        raw[a != 0] = a[a != 0]
    mapped = np.full_like(raw, IGNORE)
    for code, index in WORLDCOVER_MAP.items():
        mapped[raw == code] = index
    return mapped


def get_independent_labels(label_path, crs, transform, width, height):
    """Only accepts independent vector polygons; class_id must be 0..4."""
    import geopandas as gpd
    from rasterio.features import rasterize
    frame = gpd.read_file(label_path)
    if not {'geometry','class_id'}.issubset(frame.columns):
        raise ValueError('Label polygons must contain geometry and integer class_id 0..4')
    if frame.crs is None:
        raise ValueError('Label polygons must have CRS metadata')
    if not frame['class_id'].dropna().isin(range(5)).all():
        raise ValueError('class_id values must be integers 0..4')
    frame = frame.to_crs(crs)
    return rasterize(((g, int(c)) for g,c in zip(frame.geometry, frame.class_id)
                      if g is not None and not g.is_empty),
                     out_shape=(height,width), transform=transform, fill=IGNORE,
                     dtype='uint8')


def fetch_osm(bbox, crs):
    """Uses current OSM snapshot; capture timestamp for 2021-versus-current limitation."""
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 300
    print('Downloading real OSM building footprints and highways via Overpass...', flush=True)
    try:
        frame = ox.features_from_bbox(tuple(bbox), tags={'building':True, 'highway':True})
    except Exception as exc:
        raise RuntimeError('OSM Overpass request failed. Retry later or reduce AOI size; no fake OSM data will be substituted.') from exc
    frame = frame.reset_index(drop=True).to_crs(crs)
    buildings = frame[frame.get('building', np.full(len(frame), None)).notna() if 'building' in frame.columns else np.zeros(len(frame), dtype=bool)].copy()
    roads = frame[frame.get('highway', np.full(len(frame), None)).notna() if 'highway' in frame.columns else np.zeros(len(frame), dtype=bool)].copy()
    buildings = buildings[buildings.geometry.geom_type.isin(['Polygon','MultiPolygon'])].copy()
    roads = roads[roads.geometry.geom_type.isin(['LineString','MultiLineString'])].copy()
    # invalid geometry repair; no assumptions about unrelated OSM tags
    if len(buildings):
        buildings['geometry'] = buildings.geometry.make_valid()
        buildings = buildings[buildings.geometry.geom_type.isin(['Polygon','MultiPolygon']) & (~buildings.geometry.is_empty)].copy()
        buildings = buildings.drop_duplicates(subset=['geometry'])
    return buildings, roads


def parse_height(row):
    import re
    for key, scale in [('height', 1.0), ('building:levels', 3.0)]:
        value = row.get(key)
        if value is None or str(value).lower() == 'nan':
            continue
        m = re.search(r'\d+(?:[.,]\d+)?', str(value))
        if m:
            h = float(m.group(0).replace(',', '.'))*scale
            if 1.5 <= h <= 500:
                return h, key != 'height'
    return None, True


def bearing_axial(geom, inverse_transform=None):
    """Major axis of minimum rotated rectangle in projected metres, theta radians.

    A near-square is rotationally ambiguous and excluded from orientation averages.
    """
    rect = geom.minimum_rotated_rectangle
    if rect.geom_type != 'Polygon':
        return None
    p = list(rect.exterior.coords)
    edges = [(p[i+1][0]-p[i][0], p[i+1][1]-p[i][1]) for i in range(4)]
    lengths = [sqrt(dx*dx+dy*dy) for dx,dy in edges]
    if max(lengths) < 1 or min(lengths) / max(lengths) > 0.98:
        return None
    index = int(np.argmax(lengths))
    dx,dy = edges[index]
    if inverse_transform is not None:
        from pyproj import Geod
        longitude1, latitude1 = inverse_transform.transform(*p[index])
        longitude2, latitude2 = inverse_transform.transform(*p[index+1])
        azimuth, _, _ = Geod(ellps='WGS84').inv(longitude1,latitude1,longitude2,latitude2)
        return radians(azimuth) % pi  # geographic true-north azimuth
    return atan2(dx,dy) % pi  # grid north if no CRS transformer is supplied


def _subset(frame, poly):
    if len(frame) == 0:
        return frame
    positions = frame.sindex.query(poly, predicate='intersects')
    return frame.iloc[positions]


def osm_patch_features(buildings, roads, poly, inverse_transform=None):
    area = poly.area
    bs = _subset(buildings, poly)
    rs = _subset(roads, poly)
    sinvals, cosvals, heights, building_areas = [], [], [], []
    missing_height = 0
    covered = 0.0
    for _, row in bs.iterrows():
        geom = row.geometry
        inside = geom.intersection(poly)
        if inside.is_empty: continue
        inside_area = inside.area
        if inside_area < 1: continue
        covered += inside_area
        building_areas.append(geom.area)
        theta = bearing_axial(geom, inverse_transform)
        if theta is not None:
            sinvals.append(sin(2*theta)); cosvals.append(cos(2*theta))
        h, is_estimated = parse_height(row)
        if h is not None:
            heights.append(h)
        if is_estimated: missing_height += 1
    count = len(building_areas)
    ms = float(np.mean(sinvals)) if sinvals else 0.0
    mc = float(np.mean(cosvals)) if cosvals else 0.0
    orient_dispersion = 1.0 - sqrt(ms*ms+mc*mc) if sinvals else 1.0
    road_len = sum(r.geometry.intersection(poly).length for _,r in rs.iterrows())
    return np.array([
        ms, mc, orient_dispersion, count/(area/1e6), min(1.0, covered/area),
        np.mean(heights) if heights else 0.,
        missing_height/count if count else 1.,
        (road_len/1000.)/(area/1e6),
        np.mean(building_areas) if count else 0.,
        log1p(count)
    ], dtype=np.float32), {'buildings':count, 'missing_or_estimated_height':missing_height,
                          'orientable_buildings':len(sinvals)}


def prepare(cfg, labels_file=None):
    import rasterio
    from shapely.geometry import box
    from rasterio.windows import Window, bounds as window_bounds
    from .utils import seed_all
    seed_all(int(cfg['seed']))
    if cfg['source_year'] != 2021 and labels_file is None:
        raise ValueError('The bundled WorldCover benchmark is for 2021 only.')
    out = Path(cfg['output_dir'])
    patches = out / 'patches'
    patches.mkdir(parents=True, exist_ok=True)
    existing = list(patches.glob('*.npz'))
    if existing:
        raise RuntimeError(f'{len(existing)} patch files already exist in {patches}. Use a fresh output_dir to avoid mixing datasets.')
    crs, transform, width, height = projected_grid(cfg['bbox'], cfg['resolution_m'], cfg['patch_size'])
    images, valid, scenes = acquire_sentinel(cfg, crs, transform, width, height)
    masks = (get_independent_labels(labels_file, crs, transform, width, height) if labels_file
             else get_worldcover(cfg, crs, transform, width, height))
    buildings, roads = fetch_osm(cfg['bbox'], crs)
    from pyproj import Transformer
    inverse = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    ps = cfg['patch_size']; records=[]; groups=[]
    totals = {'buildings':0, 'missing_or_estimated_height':0, 'orientable_buildings':0}
    for r in range(height//ps):
        for c in range(width//ps):
            ys=slice(r*ps,(r+1)*ps); xs=slice(c*ps,(c+1)*ps)
            m = masks[ys,xs]
            present = m[m != IGNORE]
            if len(present)/m.size < cfg['min_label_coverage']:
                continue
            hist = np.bincount(present, minlength=5)
            label = int(np.argmax(hist))
            if hist[label]/len(present) < cfg['min_class_dominance']:
                continue
            if (1-valid[:,ys,xs].mean(axis=(1,2))).max() > cfg['max_patch_cloud_fraction']:
                continue
            bounds = window_bounds(Window(c*ps,r*ps,ps,ps), transform)
            feats, counts = osm_patch_features(buildings, roads, box(*bounds), inverse)
            for key in totals: totals[key] += counts[key]
            stem = f'patch_{r:04d}_{c:04d}'
            np.savez_compressed(patches / (stem+'.npz'), images=images[:,:,ys,xs],
                                mask=m, label=np.uint8(label), osm=feats)
            records.append({'id':stem, 'row':r, 'col':c, 'class_id':label,
                            'known_pixel_fraction':round(len(present)/m.size,4), **counts})
            block = int(cfg.get('geographic_block_patches',3))
            groups.append(f'{r//block}_{c//block}')
    if len(records) < 30:
        raise RuntimeError(f'Only {len(records)} patches accepted; increase AOI, widen dates or relax cloud filtering. Incomplete patches remain in {out}; remove them before retrying.')
    split = grouped_split(groups, seed=cfg['seed'])
    save_json(out/'split.json', split)
    save_json(out/'feature_stats.json', compute_feature_stats(out,split))
    class_names = ['residential','commercial','industrial','agricultural','recreational'] if labels_file else WORLDCOVER_CLASSES
    save_json(out/'metadata.json', {'dataset_type': 'independent_five_class_land_use_polygons' if labels_file else 'public_five_class_worldcover_land_cover',
        'label_source': str(labels_file) if labels_file else 'ESA WorldCover 2021 v200',
        'label_source_is_independent_of_osm_features': labels_file is None, # external labels require user verification
        'classes': class_names, 'bands':cfg['bands'],'n_dates':len(scenes), 'patch_size':ps,
        'study_area':cfg['study_area'], 'bbox':cfg['bbox'], 'crs':str(crs), 'transform_gdal':list(transform.to_gdal()), 'resolution_m':cfg['resolution_m'],
        'sentinel_scenes':scenes, 'osm_snapshot_utc':datetime.now(timezone.utc).isoformat(),
        'osm_height_stats_for_accepted_patches':totals, 'n_patches':len(records),
        'n_per_class':{class_names[i]:int(sum(x['class_id']==i for x in records)) for i in range(5)},
        'split_sizes':{k:len(v) for k,v in split.items()},
        'IMPORTANT': 'Satellite observations are 2021; OSM is the acquisition-time snapshot. WorldCover classes are LAND COVER, not the paper\'s urban LAND USE.'})
    save_json(out/'records.json',records)
    print(f'Prepared {len(records)} genuine patches in {out}. Split: '+str({k:len(v) for k,v in split.items()}),flush=True)
    return out
