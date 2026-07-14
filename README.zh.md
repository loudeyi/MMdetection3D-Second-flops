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

- 从 config 的 `middle_encoder.sparse_shape` 提取稀疏网格尺寸来生成合法的体素坐标
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

- coors 格式 `(batch_idx, z_idx, y_idx, x_idx)`，与 SparseEncoder 文档一致

### 3. 同时修复了 `point` 和 `image` modality

同样改用 `input_constructor` 构建正确的 dict 输入格式，避免裸 tensor 传入模型导致类型错误。

## 测试命令

```bash
conda activate openmmlab
cd /path/to/mmdetection3d
python tools/analysis_tools/get_flops.py \
    configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py \
    --shape 40000 5 4 --modality voxel
```

## 测试结果

```
==============================
Input shape: (40000, 5, 4)
Flops: 70.07 GFLOPs
Params: 5.33 M
==============================
```

模型各模块 FLOPs/Params 分布：

| 模块 | FLOPs | Params |
|------|-------|--------|
| SparseEncoder (middle_encoder) | 0.256 GFLOPs | 0.001 M |
| SECOND (backbone) | 65.0 GFLOPs | 4.28 M |
| SECONDFPN (neck) | 3.51 GFLOPs | 0.30 M |
| Anchor3DHead (bbox_head) | 1.30 GFLOPs | 0.037 M |

## 使用说明

### voxel modality

用于体素输入的模型（如 SECOND、VoxelNet、CenterPoint 等）：

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality voxel \
    --shape <num_voxels> <max_points> <num_features>
```

- `num_voxels`: 体素数量
- `max_points`: 每个体素内最大点数
- `num_features`: 每个点的特征数（如 x, y, z, intensity = 4）

### point modality

用于点云直接输入的模型：

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality point \
    --shape <num_points> <num_features>
```

### image modality

用于图像输入的模型：

```bash
python tools/analysis_tools/get_flops.py <config> \
    --modality image \
    --shape <height> <width>
```
