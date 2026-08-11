import argparse
import logging
import os
import sys

os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')


def build_parser():
    parser = argparse.ArgumentParser(
        description='Export Fast-FoundationStereo as one ONNX with an FFSGWCVolume TensorRT plugin node')
    parser.add_argument('--model_dir', type=str,
                        default=f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth')
    parser.add_argument('--save_path', type=str, default=f'{code_dir}/../output_plugin_onnx')
    parser.add_argument('--height', type=int, default=608)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--valid_iters', type=int, default=8)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--onnx_name', type=str, default='fast_foundationstereo_plugin.onnx')
    return parser


if any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    build_parser().print_help()
    sys.exit(0)

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from torch.onnx import symbolic_helper

from core.foundation_stereo import TrtFeatureRunner, TrtPostRunner


class FFSGWCVolumeOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features_left_04, features_right_04, max_disp, cv_group, normalize):
        # ONNX export only needs a tensor with the correct static shape here.
        # symbolic() emits the TensorRT plugin node that computes the real volume.
        batch, _, height, width = features_left_04.shape
        return features_left_04.new_zeros(
            (batch, int(cv_group), int(max_disp), height, width)
        )

    @staticmethod
    def symbolic(g, features_left_04, features_right_04, max_disp, cv_group, normalize):
        def as_int(value):
            if isinstance(value, int):
                return value
            return symbolic_helper._parse_arg(value, 'i')

        max_disp = as_int(max_disp)
        cv_group = as_int(cv_group)
        normalize = as_int(normalize)
        out = g.op(
            'FFSGWCVolume',
            features_left_04,
            features_right_04,
            max_disp_i=int(max_disp),
            cv_group_i=int(cv_group),
            normalize_i=int(normalize),
        )
        sizes = features_left_04.type().sizes()
        if sizes is not None and len(sizes) == 4:
            out.setType(features_left_04.type().with_sizes(
                [sizes[0], int(cv_group), int(max_disp), sizes[2], sizes[3]]))
        return out


class FastFoundationStereoPluginOnnx(nn.Module):
    def __init__(self, model, max_disp_levels, cv_group, normalize):
        super().__init__()
        self.feature_runner = TrtFeatureRunner(model)
        self.post_runner = TrtPostRunner(model)
        self.max_disp_levels = int(max_disp_levels)
        self.cv_group = int(cv_group)
        self.normalize = int(bool(normalize))

    @torch.no_grad()
    def forward(self, left, right):
        features_left_04, features_left_08, features_left_16, features_left_32, features_right_04, stem_2x = (
            self.feature_runner(left, right)
        )
        gwc_volume = FFSGWCVolumeOp.apply(
            features_left_04,
            features_right_04,
            self.max_disp_levels,
            self.cv_group,
            self.normalize,
        )
        disp = self.post_runner(
            features_left_04.float(),
            features_left_08.float(),
            features_left_16.float(),
            features_left_32.float(),
            features_right_04.float(),
            stem_2x.float(),
            gwc_volume.float(),
        )
        return disp


if __name__ == '__main__':
    args = build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    assert args.height % 32 == 0 and args.width % 32 == 0, 'height and width must be divisible by 32'
    os.makedirs(args.save_path, exist_ok=True)
    torch.autograd.set_grad_enabled(False)

    logging.info('Loading model: %s', args.model_dir)
    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.valid_iters
    model.cuda().eval()

    cv_group = int(getattr(model, 'cv_group', getattr(model.args, 'cv_group', 8)))
    normalize = bool(getattr(model.args, 'normalize', True))
    wrapper = FastFoundationStereoPluginOnnx(
        model,
        max_disp_levels=args.max_disp // 4,
        cv_group=cv_group,
        normalize=normalize,
    ).cuda().eval()

    left = torch.randn(1, 3, args.height, args.width, device='cuda').float() * 255
    right = torch.randn(1, 3, args.height, args.width, device='cuda').float() * 255

    onnx_name = args.onnx_name if args.onnx_name.endswith('.onnx') else f'{args.onnx_name}.onnx'
    onnx_path = os.path.join(args.save_path, onnx_name)
    logging.info('Exporting plugin ONNX: %s', onnx_path)

    torch.onnx.export(
        wrapper,
        (left, right),
        onnx_path,
        opset_version=17,
        input_names=['left', 'right'],
        output_names=['disp'],
        do_constant_folding=True,
        dynamo=False,
    )

    cfg = OmegaConf.to_container(model.args)
    cfg['image_size'] = [args.height, args.width]
    cfg['cv_group'] = cv_group
    cfg['normalize'] = normalize
    with open(os.path.join(args.save_path, 'onnx.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)

    logging.info('ONNX model: %s', onnx_path)
    logging.info('Config    : %s', os.path.join(args.save_path, 'onnx.yaml'))
    logging.info('Build with:')
    logging.info('  cpp/build/ffs_build_single_engine %s %s',
                 onnx_path, os.path.join(args.save_path, 'fast_foundationstereo.engine'))
