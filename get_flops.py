# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope

from mmdet3d.registry import MODELS

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[40000, 4],
        help='input point cloud size')
    parser.add_argument(
        '--modality',
        type=str,
        default='point',
        choices=['point', 'image', 'multi', 'voxel'],
        help='input data modality')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # Try to get the device of the model for creating input tensors
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    if args.modality == 'point':
        assert len(args.shape) == 2, \
            f'invalid input shape for point modality, ' \
            f'expected 2 values (num_points, num_features), ' \
            f'but got {len(args.shape)}'
        input_shape = tuple(args.shape)

        def input_constructor(input_shape):
            N, C = input_shape
            points = torch.randn(N, C, device=device)
            return {'inputs': {'points': [points]}}

    elif args.modality == 'image':
        if len(args.shape) == 1:
            input_shape = (3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (3, ) + tuple(args.shape)
        else:
            raise ValueError(f'invalid input shape for image modality, '
                             f'expected 1 or 2 values, '
                             f'but got {len(args.shape)}')

        def input_constructor(input_shape):
            img = torch.randn(1, *input_shape, device=device)
            return {'inputs': img[0]}

    elif args.modality == 'voxel':
        assert len(args.shape) == 3, \
            f'invalid input shape for voxel modality, ' \
            f'expected 3 values (num_voxels, max_points, num_features), ' \
            f'but got {len(args.shape)}'
        input_shape = tuple(args.shape)

        # Extract sparse shape from model config for generating valid coors
        middle_encoder_cfg = cfg.model.get('middle_encoder', None)
        if middle_encoder_cfg is None:
            raise ValueError(
                'Cannot find middle_encoder in model config, '
                'which is required for voxel modality.')
        sparse_shape = middle_encoder_cfg.get('sparse_shape', None)
        if sparse_shape is None:
            raise ValueError(
                'Cannot find sparse_shape in model config middle_encoder, '
                'which is required for generating valid voxel coordinates.')

        def input_constructor(input_shape):
            M, T, C = input_shape
            voxels = torch.randn(M, T, C, device=device)
            num_points = torch.randint(1, T + 1, (M,), device=device)
            # coors format: (batch_idx, z_idx, y_idx, x_idx)
            coors = torch.zeros(M, 4, dtype=torch.int32, device=device)
            coors[:, 0] = 0  # batch_idx, always 0 for single-batch FLOPs calc
            coors[:, 1] = torch.randint(0, sparse_shape[0], (M,),
                                        device=device)  # z
            coors[:, 2] = torch.randint(0, sparse_shape[1], (M,),
                                        device=device)  # y
            coors[:, 3] = torch.randint(0, sparse_shape[2], (M,),
                                        device=device)  # x
            return {
                'inputs': {
                    'voxels': {
                        'voxels': voxels,
                        'num_points': num_points,
                        'coors': coors
                    }
                }
            }

    elif args.modality == 'multi':
        raise NotImplementedError(
            'FLOPs counter is currently not supported for models with '
            'multi-modality input')

    flops, params = get_model_complexity_info(
        model, input_shape, input_constructor=input_constructor)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
