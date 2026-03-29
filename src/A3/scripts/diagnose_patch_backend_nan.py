from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.data.wsi_manifest import WSITileDataset  # noqa: E402
from bach_mil.runtime.patch_backends import create_patch_backend  # noqa: E402
from bach_mil.runtime.submission_defaults import get_default_backend_artifacts, get_task_defaults  # noqa: E402
from bach_mil.utils.io import save_json  # noqa: E402


def _finite_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    stats = {
        'shape': list(arr.shape),
        'finite_ratio': float(finite.mean()),
        'nan_count': int(np.isnan(arr).sum()),
        'inf_count': int(np.isinf(arr).sum()),
    }
    if finite.any():
        finite_vals = arr[finite]
        stats.update(
            {
                'min': float(finite_vals.min()),
                'max': float(finite_vals.max()),
                'mean': float(finite_vals.mean()),
                'std': float(finite_vals.std()),
            }
        )
    else:
        stats.update({'min': None, 'max': None, 'mean': None, 'std': None})
    return stats


def _make_real_wsi_batch(manifest_csv: Path, *, batch_size: int, input_size: int) -> tuple[torch.Tensor, dict]:
    df = pd.read_csv(manifest_csv).head(int(batch_size)).copy()
    tf = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    ds = WSITileDataset(df, transform=tf, return_label=False)
    batch = torch.stack([ds[i]['image'] for i in range(len(ds))], dim=0)
    meta = {
        'sample_source': 'real_wsi_tiles',
        'manifest_csv': str(manifest_csv),
        'slide_ids': sorted(df.slide_id.astype(str).unique().tolist()),
        'rows': int(len(df)),
    }
    return batch, meta


def _make_random_batch(*, batch_size: int, input_size: int, seed: int, low: float, high: float) -> tuple[torch.Tensor, dict]:
    rng = np.random.default_rng(seed)
    arr = rng.uniform(low=float(low), high=float(high), size=(int(batch_size), 3, int(input_size), int(input_size))).astype(np.float32)
    meta = {
        'sample_source': 'random_uniform',
        'seed': int(seed),
        'low': float(low),
        'high': float(high),
    }
    return torch.from_numpy(arr), meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi', choices=['wsi', 'photos'])
    parser.add_argument('--backend', type=str, required=True, choices=['pytorch', 'onnx', 'om'])
    parser.add_argument('--report_json', type=str, required=True)
    parser.add_argument('--sample_source', type=str, default='real_wsi_tiles', choices=['real_wsi_tiles', 'random_uniform'])
    parser.add_argument('--manifest_csv', type=str, default='data/BACH/derived/split/wsi_train_tiles_L1_s448_mt40.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--random_low', type=float, default=-2.5)
    parser.add_argument('--random_high', type=float, default=2.5)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--onnx_path', type=str, default=None)
    parser.add_argument('--om_path', type=str, default=None)
    parser.add_argument('--meta_json', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--backbone_pool', type=str, default=None)
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--torch_device', type=str, default='cpu')
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
    input_size = int(args.input_size or defaults['input_size'])

    if args.sample_source == 'real_wsi_tiles':
        batch, sample_meta = _make_real_wsi_batch(Path(args.manifest_csv), batch_size=args.batch_size, input_size=input_size)
    else:
        batch, sample_meta = _make_random_batch(
            batch_size=args.batch_size,
            input_size=input_size,
            seed=args.seed,
            low=args.random_low,
            high=args.random_high,
        )

    backend = create_patch_backend(
        backend=args.backend,
        ckpt_path=ckpt,
        model_name=str(args.model_name or defaults['model_name']),
        backbone_pool=str(args.backbone_pool or defaults['backbone_pool']),
        backbone_init_values=args.backbone_init_values if args.backbone_init_values is not None else defaults['backbone_init_values'],
        default_classes=defaults['labels'],
        input_size=input_size,
        onnx_path=onnx_path,
        om_path=om_path,
        meta_json=meta_json,
        device_id=args.device_id,
        torch_device=args.torch_device,
        om_execution_mode=args.om_execution_mode,
        om_host_io_mode=args.om_host_io_mode,
        om_output_mode=args.om_output_mode,
    )
    try:
        out = backend.predict_batch(batch)
    finally:
        backend.close()

    batch_np = batch.detach().cpu().numpy()
    summary = {
        'task': str(args.task),
        'backend': str(args.backend),
        'backend_actual': getattr(backend, 'backend_name', str(args.backend)),
        'sample_meta': sample_meta,
        'batch_stats': _finite_stats(batch_np),
        'features_stats': _finite_stats(out.features),
        'logits_stats': _finite_stats(out.logits),
        'probs_stats': _finite_stats(out.probs),
        'om_execution_mode': str(args.om_execution_mode),
        'om_host_io_mode': str(args.om_host_io_mode),
        'om_output_mode': str(args.om_output_mode),
        'ckpt': str(ckpt),
        'onnx_path': str(onnx_path),
        'om_path': str(om_path),
        'meta_json': str(meta_json),
    }
    save_json(summary, args.report_json)
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
