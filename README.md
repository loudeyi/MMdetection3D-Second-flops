# MMdetection3D-Second-flops

mmdet3d-flops-tools extends MMDetection3D's FLOPs calculator with native voxel modality support, automatically constructing dictionary inputs (voxels, num_points, coors) from config's sparse_shape for voxel-based detectors like SECOND and PointPillars.

## Root Cause

`get_model_complexity_info` calls `model(input_tensor)`, but mmdet3d models (including VoxelNet/SECOND) expect dict-format input: `model(inputs={'voxels': {...}})`. The previous code only created a bare tensor and passed it to the model, which failed for voxel-based models.

## Changes

### 1. Added `voxel` modality (line 29)

`--modality` choices now include `'voxel'`:

```python
choices=['point', 'image', 'multi', 'voxel']
```

### 2. Voxel input handling (lines 90-130)

Accepts 3 shape parameters `(num_voxels, max_points, num_features)`:

- Extracts `sparse_shape` from config's `middle_encoder` to generate valid voxel coordinates
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

- Coors format: `(batch_idx, z_idx, y_idx, x_idx)`, consistent with the SparseEncoder documentation

### 3. Also fixed `point` and `image` modalities

Both now use `input_constructor` to build the correct dict input format, avoiding type errors from passing bare tensors.

## Test Command

```bash
conda activate openmmlab
cd /home/hzh/mmdetection3d
python tools/analysis_tools/get_flops.py \
    configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py \
    --shape 40000 5 4 --modality voxel
```

## Test Results

```
==============================
Input shape: (40000, 5, 4)
Flops: 70.07 GFLOPs
Params: 5.33 M
==============================
```

Per-module FLOPs/Params breakdown:

| Module | FLOPs | Params |
|--------|-------|--------|
| SparseEncoder (middle_encoder) | 0.256 GFLOPs | 0.001 M |
| SECOND (backbone) | 65.0 GFLOPs | 4.28 M |
| SECONDFPN (neck) | 3.51 GFLOPs | 0.30 M |
| Anchor3DHead (bbox_head) | 1.30 GFLOPs | 0.037 M |

## Usage

### voxel modality

For voxel-based models (e.g., SECOND, VoxelNet, CenterPoint):

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality voxel \
    --shape <num_voxels> <max_points> <num_features>
```

- `num_voxels`: Number of voxels
- `max_points`: Maximum points per voxel
- `num_features`: Features per point (e.g., x, y, z, intensity = 4)

### point modality

For point-cloud models:

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality point \
    --shape <num_points> <num_features>
```

### image modality

For image-based models:

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality image \
    --shape <height> <width>
```
