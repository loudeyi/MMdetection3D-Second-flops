🌐 [English](README.md) | [简体中文](README.zh.md)

---
# MMdetection3D-Second-flops

mmdet3d-flops-tools extends MMDetection3D's FLOPs calculator with native voxel modality support, automatically constructing dictionary inputs (voxels, num_points, coors) from config's sparse_shape for voxel-based detectors like SECOND and PointPillars.

## Root Cause

`get_model_complexity_info` calls `model(input_tensor)`, but mmdet3d models (including VoxelNet/SECOND) expect dict-format input: `model(inputs={'voxels': {...}})`. The previous code only created a bare tensor and passed it to the model, which failed for voxel-based models.

## Changes

### 1. Added `voxel` modality

`--modality` choices now include `'voxel'`:

```python
choices=['point', 'image', 'multi', 'voxel']
```

### 2. Voxel input handling

Accepts 3 shape parameters `(num_voxels, max_points, num_features)`:

- Automatically detects the middle encoder type from config:
  - **SparseEncoder** → uses `sparse_shape` (e.g. `[41, 1600, 1408]`) for 3D sparse conv voxels
  - **PointPillarsScatter** → uses `output_shape` (e.g. `[496, 432]`) for 2D BEV pillar scatter
- Uses an `input_constructor` closure to build the complete voxel input dict:

```python
{
    'inputs': {
        'voxels': {
            'voxels': ...,       # (M, T, C)
            'num_points': ...,   # (M,)
            'coors': ...         # (M, 4)
        }
    }
}
```

- Coors format: `(batch_idx, z_idx, y_idx, x_idx)`, consistent with the SparseEncoder / PointPillarsScatter documentation

### 3. Fixed `point` and `image` modalities

Both now use `input_constructor` to build the correct dict input format:
- `point`: returns `{'inputs': {'points': [tensor]}}`
- `image`: returns `{'inputs': {'imgs': tensor}}`

### 4. Added `--out` log output option

New `--out` argument to save the full FLOPs computation log (including per-layer breakdown) to a file, while still printing to stdout:

```bash
python tools/analysis_tools/get_flops.py <config> --modality voxel \
    --shape 40000 5 4 --out flops_log.txt
```

## Test Results

All three input modalities verified on real models:

### Voxel: SECOND

```bash
python tools/analysis_tools/get_flops.py \
    configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py \
    --shape 40000 5 4 --modality voxel
```

| Metric | Value |
|--------|-------|
| Input Shape | (40000, 5, 4) |
| FLOPs | 70.07 GFLOPs |
| Params | 5.33 M |

### Voxel: PointPillars

```bash
python tools/analysis_tools/get_flops.py \
    configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    --shape 16000 32 4 --modality voxel
```

| Metric | Value |
|--------|-------|
| Input Shape | (16000, 32, 4) |
| FLOPs | 34.72 GFLOPs |
| Params | 4.83 M |

### Point: 3DSSD

```bash
python tools/analysis_tools/get_flops.py \
    configs/3dssd/3dssd_4xb4_kitti-3d-car.py \
    --shape 16384 4 --modality point
```

| Metric | Value |
|--------|-------|
| Input Shape | (16384, 4) |
| FLOPs | 16.03 GFLOPs |
| Params | 2.51 M |

### Image: PGD

```bash
python tools/analysis_tools/get_flops.py \
    configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py \
    --shape 375 1242 --modality image
```

| Metric | Value |
|--------|-------|
| Input Shape | (3, 375, 1242) |
| FLOPs | 403.09 GFLOPs |
| Params | 54.72 M |

## Usage

### voxel modality

For voxel-based models (e.g., SECOND, VoxelNet, PointPillars, CenterPoint):

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality voxel \
    --shape <num_voxels> <max_points> <num_features>
```

- `num_voxels`: Number of voxels
- `max_points`: Maximum points per voxel
- `num_features`: Features per point (e.g., x, y, z, intensity = 4)

### point modality

For point-cloud models (e.g., 3DSSD, PointRCNN, VoteNet):

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality point \
    --shape <num_points> <num_features>
```

### image modality

For image-based models (e.g., FCOS3D, PGD, SMOKE):

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality image \
    --shape <height> <width>
```

### Save output to file

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality voxel --shape 40000 5 4 \
    --out logs/flops_second.txt
```
