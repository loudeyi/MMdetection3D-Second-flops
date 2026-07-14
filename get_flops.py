# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys

import torch
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope

from mmdet3d.registry import MODELS

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


class Tee:
    """Redirect output to multiple streams (stdout + log file)."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


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
        '--out',
        type=str,
        default=None,
        help='path to save the output log file. '
        'If not specified, output is only printed to stdout.')
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
            # Create batched image tensor (B, C, H, W)
            img = torch.randn(1, *input_shape, device=device)
            return {'inputs': {'imgs': img}}

    elif args.modality == 'voxel':
        assert len(args.shape) == 3, \
            f'invalid input shape for voxel modality, ' \
            f'expected 3 values (num_voxels, max_points, num_features), ' \
            f'but got {len(args.shape)}'
        input_shape = tuple(args.shape)

        # Extract shape info from model config for generating valid coors.
        # Different middle encoders use different shape parameter names:
        # - SparseEncoder: sparse_shape (e.g. [41, 1600, 1408] for Z, Y, X)
        # - PointPillarsScatter: output_shape (e.g. [496, 432] for Y, X)
        middle_encoder_cfg = cfg.model.get('middle_encoder', None)
        if middle_encoder_cfg is None:
            raise ValueError(
                'Cannot find middle_encoder in model config, '
                'which is required for voxel modality.')

        sparse_shape = middle_encoder_cfg.get('sparse_shape', None)
        output_shape = middle_encoder_cfg.get('output_shape', None)

        if sparse_shape is None and output_shape is None:
            raise ValueError(
                'Cannot find sparse_shape or output_shape in model config '
                'middle_encoder, which is required for generating valid '
                'voxel coordinates.')

        def input_constructor(input_shape):
            M, T, C = input_shape
            voxels = torch.randn(M, T, C, device=device)
            num_points = torch.randint(1, T + 1, (M,), device=device)
            # coors format: (batch_idx, z_idx, y_idx, x_idx)
            coors = torch.zeros(M, 4, dtype=torch.int32, device=device)
            coors[:, 0] = 0  # batch_idx, always 0 for single-batch FLOPs calc

            if sparse_shape is not None:
                # SparseEncoder: 3D sparse conv uses (z, y, x) coors
                coors[:, 1] = torch.randint(0, sparse_shape[0], (M,),
                                            device=device)  # z
                coors[:, 2] = torch.randint(0, sparse_shape[1], (M,),
                                            device=device)  # y
                coors[:, 3] = torch.randint(0, sparse_shape[2], (M,),
                                            device=device)  # x
            elif output_shape is not None:
                # PointPillarsScatter: 2D BEV scatter uses (y, x) pillar coors
                # z is always 0 (single z-slice for pillar-based models)
                coors[:, 1] = 0  # z
                coors[:, 2] = torch.randint(0, output_shape[0], (M,),
                                            device=device)  # y
                coors[:, 3] = torch.randint(0, output_shape[1], (M,),
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

    # Set up output stream: if --out is specified, tee to both stdout and file
    if args.out:
        log_file = open(args.out, 'w')
        ost = Tee(sys.stdout, log_file)
    else:
        ost = sys.stdout

    flops, params = get_model_complexity_info(
        model,
        input_shape,
        input_constructor=input_constructor,
        ost=ost)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}',
          file=ost)
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.',
          file=ost)

    if args.out:
        log_file.close()
        print(f'\nLog saved to: {args.out}')


if __name__ == '__main__':
    main()
