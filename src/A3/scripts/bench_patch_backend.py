from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.runtime.patch_backends import create_patch_backend  # noqa: E402
from bach_mil.runtime.submission_defaults import get_default_backend_artifacts, get_task_defaults  # noqa: E402
from bach_mil.utils.io import save_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi', choices=['wsi', 'photos'])
    parser.add_argument('--backend', type=str, required=True, choices=['pytorch', 'onnx', 'om'])
    parser.add_argument('--report_json', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--onnx_path', type=str, default=None)
    parser.add_argument('--om_path', type=str, default=None)
    parser.add_argument('--meta_json', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--backbone_pool', type=str, default=None)
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--bench_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--torch_device', type=str, default='cpu')
    parser.add_argument('--torch_amp_dtype', type=str, default='none', choices=['none', 'fp16', 'bf16'])
    parser.add_argument('--torch_channels_last', action='store_true')
    parser.add_argument('--torch_jit_compile', action='store_true')
    parser.add_argument('--om_execution_mode', type=str, default='sync', choices=['sync', 'async'])
    parser.add_argument('--om_host_io_mode', type=str, default='legacy', choices=['legacy', 'buffer_reuse'])
    parser.add_argument('--om_output_mode', type=str, default='both', choices=['both', 'features_only', 'logits_only'])
    args = parser.parse_args()

    defaults = get_task_defaults(args.task)
    artifacts = get_default_backend_artifacts(args.task)
    ckpt = Path(args.ckpt) if args.ckpt is not None else Path(defaults['ckpt'])
    onnx_path = Path(args.onnx_path) if args.onnx_path is not None else artifacts['onnx']
    om_path = Path(args.om_path) if args.om_path is not None else artifacts['om']
    if args.meta_json is not None:
        meta_json = Path(args.meta_json)
    else:
        meta_json = artifacts.get('om_meta_json', artifacts['meta_json']) if args.backend == 'om' else artifacts['meta_json']
    model_name = str(args.model_name or defaults['model_name'])
    backbone_pool = str(args.backbone_pool or defaults['backbone_pool'])
    backbone_init_values = args.backbone_init_values if args.backbone_init_values is not None else defaults['backbone_init_values']
    input_size = int(args.input_size or defaults['input_size'])

    rng = np.random.default_rng(args.seed)
    sample = rng.standard_normal((args.batch_size, 3, input_size, input_size), dtype=np.float32)
    sample_tensor = torch.from_numpy(sample)

    backend = create_patch_backend(
        backend=args.backend,
        ckpt_path=ckpt,
        model_name=model_name,
        backbone_pool=backbone_pool,
        backbone_init_values=backbone_init_values,
        default_classes=defaults['labels'],
        input_size=input_size,
        onnx_path=onnx_path,
        om_path=om_path,
        meta_json=meta_json,
        device_id=args.device_id,
        torch_device=args.torch_device,
        amp_dtype=args.torch_amp_dtype,
        channels_last=args.torch_channels_last,
        jit_compile=args.torch_jit_compile,
        om_execution_mode=args.om_execution_mode,
        om_host_io_mode=args.om_host_io_mode,
        om_output_mode=args.om_output_mode,
    )
    try:
        for _ in range(int(args.warmup_steps)):
            backend.predict_batch(sample_tensor)
        start = time.perf_counter()
        total_samples = 0
        for _ in range(int(args.bench_steps)):
            out = backend.predict_batch(sample_tensor)
            total_samples += int(len(out.features))
        elapsed = time.perf_counter() - start
    finally:
        backend.close()

    summary = {
        'task': str(args.task),
        'backend': str(args.backend),
        'batch_size': int(args.batch_size),
        'warmup_steps': int(args.warmup_steps),
        'bench_steps': int(args.bench_steps),
        'elapsed_sec': float(elapsed),
        'total_samples': int(total_samples),
        'patches_per_sec': float(total_samples / max(elapsed, 1e-12)),
        'sec_per_patch': float(elapsed / max(total_samples, 1)),
        'torch_amp_dtype': str(args.torch_amp_dtype),
        'torch_device': str(args.torch_device),
        'om_execution_mode': str(args.om_execution_mode),
        'om_host_io_mode': str(args.om_host_io_mode),
        'om_output_mode': str(args.om_output_mode),
        'ckpt': str(ckpt),
        'onnx_path': str(onnx_path),
        'om_path': str(om_path),
        'meta_json': str(meta_json),
        'backend_actual': getattr(backend, 'backend_name', str(args.backend)),
    }
    save_json(summary, args.report_json)
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
