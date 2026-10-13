# Implementation verification record — 2026-10-13

## Completed checks

1. `python -m compileall -q urbanvista` — completed.
2. Editable installation: `pip install -e . --no-deps --no-build-isolation` — completed.
3. CLI discovery: `urbanvista --help` — exposes `prepare`, `run`, `predict`, `smoke-test`.
4. `PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m pytest -q` — **7 passed**.
5. `urbanvista smoke-test --workdir /mnt/data/urbanvista_smoke_validation2` — completed a full synthetic-only training/validation/test pass and wrote checkpoints, figures and metrics.
6. `urbanvista predict --checkpoint ... --patch ... --dataset-root ...` — saved prediction JSON and segmentation PNG using the smoke checkpoint.
7. The reprojected single-band raster reader was exercised on a local GeoTIFF. Full dataset-construction bookkeeping was tested using mocked offline Sentinel, WorldCover and OSM input arrays.
8. The WorldCover S3 URL scheme, Planetary Computer signing pattern and OSMnx bbox API were cross-checked with published service documentation.

## Not verified in this environment

- Actual live STAC, S3 and Overpass acquisition: runtime networking fails DNS resolution. These services may be rate-limited or require a changed query at execution time; run `urbanvista prepare` on an internet-connected workstation before representing them as tested.
- Multi-day 100-epoch training or exact results from the UrbanVista manuscript. No original raw dataset, fitted weights, model hyperparameter logs or original split files were supplied. This repository is independently developed.
- Optional `timm` ViT/Swin execution; the current local runtime did not have that optional dependency. Install `pip install -e '.[all]'` and verify in the target GPU environment.
- The paper's five residential/commercial/industrial/agricultural/recreational land-use labels: the automated benchmark uses the distinctly different ESA WorldCover land-cover classes. Original experiments cannot be validated without independent original labels.

The synthetic-only smoke metrics are not scientific results and must not be pasted into a manuscript.
