from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.runtime.patch_backends import cosine_similarity_mean, create_patch_backend  # noqa: E402
from bach_mil.runtime.submission_defaults import get_default_backend_artifacts, get_task_defaults  # noqa: E402
from bach_mil.utils.io import save_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi', choices=['wsi', 'photos'])
    parser.add_argument('--after_backend', type=str, required=True, choices=['onnx', 'om'])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--onnx_path', type=str, default=None)
    parser.add_argument('--om_path', type=str, default=None)
    parser.add_argument('--meta_json', type=str, default=None)
    parser.add_argument('--report_json', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--backbone_pool', type=str, default=None)
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--om_execution_mode', type=str, default='sync', choices=['sync', 'async'])
    parser.add_argument('--om_host_io_mode', type=str, default='legacy', choices=['legacy', 'buffer_reuse'])
    parser.add_argument('--om_output_mode', type=str, default='both', choices=['both', 'features_only', 'logits_only'])
    args = parser.parse_args()

    defaults = get_task_defaults(args.task)
    artifacts = get_default_backend_artifacts(args.task)
    ckpt_path = Path(args.ckpt) if args.ckpt is not None else Path(defaults['ckpt'])
    onnx_path = Path(args.onnx_path) if args.onnx_path is not None else artifacts['onnx']
    om_path = Path(args.om_path) if args.om_path is not None else artifacts['om']
    if args.meta_json is not None:
        meta_json = Path(args.meta_json)
    else:
        meta_json = artifacts.get('om_meta_json', artifacts['meta_json']) if args.after_backend == 'om' else artifacts['meta_json']
    if args.report_json is not None:
        report_json = Path(args.report_json)
    else:
        report_json = artifacts['onnx_parity_json'] if args.after_backend == 'onnx' else artifacts['om_parity_json']
    model_name = str(args.model_name or defaults['model_name'])
    backbone_pool = str(args.backbone_pool or defaults['backbone_pool'])
    backbone_init_values = args.backbone_init_values if args.backbone_init_values is not None else defaults['backbone_init_values']
    input_size = int(args.input_size or defaults['input_size'])

    rng = np.random.default_rng(args.seed)
    sample = rng.standard_normal((args.batch_size, 3, input_size, input_size), dtype=np.float32)
    sample_tensor = torch.from_numpy(sample)

    before_backend = create_patch_backend(
        backend='pytorch',
        ckpt_path=ckpt_path,
        model_name=model_name,
        backbone_pool=backbone_pool,
        backbone_init_values=backbone_init_values,
        default_classes=defaults['labels'],
        input_size=input_size,
        torch_device='cpu',
    )
    after_backend = create_patch_backend(
        backend=args.after_backend,
        ckpt_path=ckpt_path,
        model_name=model_name,
        backbone_pool=backbone_pool,
        backbone_init_values=backbone_init_values,
        default_classes=defaults['labels'],
        input_size=input_size,
        onnx_path=onnx_path,
        om_path=om_path,
        meta_json=meta_json,
        device_id=args.device_id,
        om_execution_mode=args.om_execution_mode,
        om_host_io_mode=args.om_host_io_mode,
        om_output_mode=args.om_output_mode,
    )
    try:
        before_out = before_backend.predict_batch(sample_tensor)
        after_out = after_backend.predict_batch(sample_tensor)
    finally:
        before_backend.close()
        after_backend.close()

    summary = {
        'task': str(args.task),
        'after_backend': str(args.after_backend),
        'status': 'ok',
        'feature_cosine_mean': cosine_similarity_mean(before_out.features, after_out.features),
        'logit_cosine_mean': cosine_similarity_mean(before_out.logits, after_out.logits),
        'feature_max_abs_diff': float(np.max(np.abs(before_out.features - after_out.features))),
        'feature_mean_abs_diff': float(np.mean(np.abs(before_out.features - after_out.features))),
        'logit_max_abs_diff': float(np.max(np.abs(before_out.logits - after_out.logits))),
        'logit_mean_abs_diff': float(np.mean(np.abs(before_out.logits - after_out.logits))),
        'prob_max_abs_diff': float(np.max(np.abs(before_out.probs - after_out.probs))),
        'prob_mean_abs_diff': float(np.mean(np.abs(before_out.probs - after_out.probs))),
        'om_execution_mode': str(args.om_execution_mode),
        'om_host_io_mode': str(args.om_host_io_mode),
        'om_output_mode': str(args.om_output_mode),
    }
    save_json(summary, report_json)
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
