🌐 [English](README.md) | [简体中文](README.zh.md)

---
# MMdetection3D-Second-flops

mmdet3d-flops-tools 扩展了 MMDetection3D 的 FLOPs 计算器，原生支持体素（voxel）模态输入，能够从配置文件的 sparse_shape 自动构建字典输入（voxels、num_points、coors），适用于 SECOND、PointPillars 等基于体素的检测器。

## 问题根因

`get_model_complexity_info` 会调用 `model(input_tensor)`，但 mmdet3d 的模型（包括 VoxelNet/SECOND）期望的输入格式是 dict：`model(inputs={'voxels': {...}})`。之前的代码仅仅创建一个裸 tensor 传给模型，无法适配体素输入模型。

## 修改点

### 1. 新增 `voxel` modality

`--modality` choices 增加 `'voxel'`：

```python
choices=['point', 'image', 'multi', 'voxel']
```

### 2. voxel 输入处理

接受3个 shape 参数 `(num_voxels, max_points, num_features)`：

- 自动检测 config 中的 middle_encoder 类型：
  - **SparseEncoder** → 使用 `sparse_shape`（如 `[41, 1600, 1408]`）生成 3D 稀疏卷积的体素坐标
  - **PointPillarsScatter** → 使用 `output_shape`（如 `[496, 432]`）生成 2D BEV pillar 坐标
- 通过 `input_constructor` 闭包构建完整的 voxel 输入 dict：

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

- coors 格式 `(batch_idx, z_idx, y_idx, x_idx)`，与 SparseEncoder / PointPillarsScatter 文档一致

### 3. 同时修复了 `point` 和 `image` modality

同样改用 `input_constructor` 构建正确的 dict 输入格式：
- `point`：返回 `{'inputs': {'points': [tensor]}}`
- `image`：返回 `{'inputs': {'imgs': tensor}}`

### 4. 新增 `--out` 日志输出选项

新增 `--out` 参数，可将完整的 FLOPs 计算日志（含逐层明细）输出到文件，同时保留终端打印：

```bash
python tools/analysis_tools/get_flops.py <config> --modality voxel \
    --shape 40000 5 4 --out flops_log.txt
```

## 测试结果

三种输入模态均在真实模型上验证通过：

### Voxel: SECOND

```bash
python tools/analysis_tools/get_flops.py \
    configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py \
    --shape 40000 5 4 --modality voxel
```

| 指标 | 数值 |
|------|------|
| Input Shape | (40000, 5, 4) |
| FLOPs | 70.07 GFLOPs |
| Params | 5.33 M |

### Voxel: PointPillars

```bash
python tools/analysis_tools/get_flops.py \
    configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
    --shape 16000 32 4 --modality voxel
```

| 指标 | 数值 |
|------|------|
| Input Shape | (16000, 32, 4) |
| FLOPs | 34.72 GFLOPs |
| Params | 4.83 M |

### Point: 3DSSD

```bash
python tools/analysis_tools/get_flops.py \
    configs/3dssd/3dssd_4xb4_kitti-3d-car.py \
    --shape 16384 4 --modality point
```

| 指标 | 数值 |
|------|------|
| Input Shape | (16384, 4) |
| FLOPs | 16.03 GFLOPs |
| Params | 2.51 M |

### Image: PGD

```bash
python tools/analysis_tools/get_flops.py \
    configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py \
    --shape 375 1242 --modality image
```

| 指标 | 数值 |
|------|------|
| Input Shape | (3, 375, 1242) |
| FLOPs | 403.09 GFLOPs |
| Params | 54.72 M |

## 使用说明

### voxel modality

用于体素输入的模型（如 SECOND、VoxelNet、PointPillars、CenterPoint 等）：

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality voxel \
    --shape <num_voxels> <max_points> <num_features>
```

- `num_voxels`: 体素数量
- `max_points`: 每个体素内最大点数
- `num_features`: 每个点的特征数（如 x, y, z, intensity = 4）

### point modality

用于点云直接输入的模型（如 3DSSD、PointRCNN、VoteNet 等）：

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality point \
    --shape <num_points> <num_features>
```

### image modality

用于图像输入的模型（如 FCOS3D、PGD、SMOKE 等）：

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality image \
    --shape <height> <width>
```

### 保存日志到文件

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality voxel --shape 40000 5 4 \
    --out logs/flops_second.txt
```
