from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

# No OSM land-use attribute enters the features: prevents leakage from land-use labels.
FEATURE_NAMES = [
    "mean_sin_2theta", "mean_cos_2theta", "orientation_dispersion",
    "buildings_per_km2", "building_footprint_fraction", "mean_height_m",
    "height_missing_fraction", "road_km_per_km2", "mean_building_area_m2",
    "building_count_log1p",
]
WORLDCOVER_MAP = {10: 0, 30: 1, 40: 2, 50: 3, 80: 4}
WORLDCOVER_CLASSES = ["tree_cover", "grassland", "cropland", "built_up", "water"]
IGNORE = 255


def grouped_split(groups, seed=42, test_size=0.15, val_size=0.15):
    groups = np.asarray(groups)
    idx = np.arange(len(groups))
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three geographic blocks are required for train/val/test.")
    first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(first.split(idx, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=seed + 1)
    itrain, ival = next(second.split(trainval, groups=groups[trainval]))
    result = {"train": idx[trainval[itrain]].tolist(), "val": idx[trainval[ival]].tolist(), "test": idx[test].tolist()}
    sets = [set(groups[result[s]]) for s in result]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]), "Geographic leakage"
    if min(map(len, result.values())) == 0:
        raise ValueError("Geographic split produced an empty partition. Enlarge the AOI.")
    return result


def compute_feature_stats(data_dir, split):
    paths = sorted(Path(data_dir, "patches").glob("*.npz"))
    arr = np.stack([np.load(paths[i])["osm"] for i in split["train"]], axis=0)
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return {"mean": mu.tolist(), "std": sd.tolist(), "names": FEATURE_NAMES}


class PatchDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = Path(root)
        with open(self.root / "split.json", encoding="utf-8") as f:
            self.split_data = json.load(f)
        with open(self.root / "feature_stats.json", encoding="utf-8") as f:
            self.stats = json.load(f)
        self.paths = sorted((self.root / "patches").glob("*.npz"))
        self.indices = self.split_data[split]
        if not self.indices:
            raise ValueError(f"No patches in {split} split")
        self.augment = augment
        self.mu = np.asarray(self.stats["mean"], dtype=np.float32)
        self.sd = np.asarray(self.stats["std"], dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        path = self.paths[self.indices[index]]
        with np.load(path) as d:
            images = d["images"].astype(np.float32)
            mask = d["mask"].astype(np.int64)
            label = int(d["label"])
            osm = d["osm"].astype(np.float32)
        if self.augment:
            if np.random.random() < .5:
                images = images[:, :, :, ::-1]; mask = mask[:, ::-1]; osm[0] *= -1  # mirror x: sin(2theta) changes sign
            if np.random.random() < .5:
                images = images[:, :, ::-1, :]; mask = mask[::-1, :]; osm[0] *= -1  # mirror y: sin(2theta) changes sign
            # Rotations would require transforming the entire geographic context.
            # They are excluded; reflections above transform axial sin(2theta).
            if np.random.random() < .5:
                images = np.clip(images * np.random.uniform(.95, 1.05), 0, 1)
        osm = np.clip((osm-self.mu)/self.sd, -10, 10)
        return {
            "images": torch.from_numpy(images.copy()),
            "osm": torch.from_numpy(osm.copy()),
            "label": torch.tensor(label, dtype=torch.long),
            "mask": torch.from_numpy(mask.copy()),
            "sample": path.stem,
        }
