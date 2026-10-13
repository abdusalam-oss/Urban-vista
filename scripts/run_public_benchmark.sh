#!/usr/bin/env bash
set -euo pipefail
urbanvista prepare --config configs/berlin_worldcover_2021.yaml
urbanvista run --config configs/berlin_worldcover_2021.yaml --models ablations
urbanvista run --config configs/berlin_worldcover_2021.yaml --models unet segnet deeplabv3plus vit swin
