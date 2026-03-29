from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import onnx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.runtime.patch_backends import (  # noqa: E402
    PatchExportWrapper,
    cosine_similarity_mean,
    create_patch_backend,
    load_patch_model_from_ckpt,
)
from bach_mil.runtime.submission_defaults import get_default_backend_artifacts, get_task_defaults  # noqa: E402
from bach_mil.utils.io import ensure_dir, save_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi', choices=['wsi', 'photos'])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--out_onnx', type=str, default=None)
    parser.add_argument('--meta_json', type=str, default=None)
    parser.add_argument('--report_json', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--backbone_pool', type=str, default=None)
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    defaults = get_task_defaults(args.task)
    artifacts = get_default_backend_artifacts(args.task)
    ckpt_path = Path(args.ckpt) if args.ckpt is not None else Path(defaults['ckpt'])
    out_onnx = Path(args.out_onnx) if args.out_onnx is not None else artifacts['onnx']
    meta_json = Path(args.meta_json) if args.meta_json is not None else artifacts['meta_json']
    report_json = Path(args.report_json) if args.report_json is not None else artifacts['onnx_export_json']
    model_name = str(args.model_name or defaults['model_name'])
    backbone_pool = str(args.backbone_pool or defaults['backbone_pool'])
    backbone_init_values = args.backbone_init_values if args.backbone_init_values is not None else defaults['backbone_init_values']
    input_size = int(args.input_size or defaults['input_size'])

    ensure_dir(out_onnx.parent)
    ensure_dir(meta_json.parent)
    ensure_dir(report_json.parent)
    torch.set_num_threads(1)

    model, classes, meta = load_patch_model_from_ckpt(
        ckpt_path=ckpt_path,
        model_name=model_name,
        backbone_pool=backbone_pool,
        backbone_init_values=backbone_init_values,
        default_classes=defaults['labels'],
    )
    model = model.cpu().eval()
    wrapper = PatchExportWrapper(model).cpu().eval()

    generator = np.random.default_rng(args.seed)
    sample = generator.standard_normal((args.batch_size, 3, input_size, input_size), dtype=np.float32)
    sample_tensor = torch.from_numpy(sample)

    start = time.time()
    torch.onnx.export(
        wrapper,
        sample_tensor,
        str(out_onnx),
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=['input'],
        output_names=['features', 'logits'],
        dynamic_axes=None,
    )
    export_elapsed = time.time() - start

    onnx_model = onnx.load(str(out_onnx))
    onnx.checker.check_model(onnx_model)

    export_meta = {
        'task': str(args.task),
        'ckpt': str(ckpt_path),
        'model_name': model_name,
        'backbone_pool': backbone_pool,
        'backbone_init_values': backbone_init_values,
        'classes': list(classes),
        'feature_dim': int(meta['feature_dim']),
        'input_name': 'input',
        'input_shape': [int(args.batch_size), 3, input_size, input_size],
        'output_order': ['features', 'logits'],
        'features_output_name': 'features',
        'logits_output_name': 'logits',
        'output_shapes': {
            'features': [int(args.batch_size), int(meta['feature_dim'])],
            'logits': [int(args.batch_size), len(classes)],
        },
        'output_dtypes': {
            'features': 'float32',
            'logits': 'float32',
        },
    }
    save_json(export_meta, meta_json)

    before_backend = create_patch_backend(
        backend='pytorch',
        ckpt_path=ckpt_path,
        model_name=model_name,
        backbone_pool=backbone_pool,
        backbone_init_values=backbone_init_values,
        default_classes=classes,
        input_size=input_size,
        torch_device='cpu',
    )
    after_backend = create_patch_backend(
        backend='onnx',
        ckpt_path=ckpt_path,
        model_name=model_name,
        backbone_pool=backbone_pool,
        backbone_init_values=backbone_init_values,
        default_classes=classes,
        input_size=input_size,
        onnx_path=out_onnx,
        meta_json=meta_json,
    )
    try:
        before_out = before_backend.predict_batch(sample_tensor)
        after_out = after_backend.predict_batch(sample_tensor)
    finally:
        before_backend.close()
        after_backend.close()

    summary = {
        'task': str(args.task),
        'status': 'ok',
        'ckpt': str(ckpt_path),
        'out_onnx': str(out_onnx),
        'meta_json': str(meta_json),
        'batch_size': int(args.batch_size),
        'input_size': int(input_size),
        'export_elapsed_sec': float(export_elapsed),
        'feature_cosine_mean': cosine_similarity_mean(before_out.features, after_out.features),
        'logit_cosine_mean': cosine_similarity_mean(before_out.logits, after_out.logits),
        'feature_max_abs_diff': float(np.max(np.abs(before_out.features - after_out.features))),
        'logit_max_abs_diff': float(np.max(np.abs(before_out.logits - after_out.logits))),
        'prob_max_abs_diff': float(np.max(np.abs(before_out.probs - after_out.probs))),
    }
    save_json(summary, report_json)
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
