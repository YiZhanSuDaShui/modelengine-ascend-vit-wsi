from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.data.label_map import PHOTO_LABELS  # noqa: E402
from bach_mil.data.wsi_manifest import WSITileDataset, list_wsi_paths, scan_slide_for_tiles  # noqa: E402
from bach_mil.runtime.patch_backends import create_patch_backend, softmax_numpy  # noqa: E402
from bach_mil.runtime.submission_defaults import get_default_backend_artifacts, get_task_defaults, get_variant_runtime_profile  # noqa: E402
from bach_mil.utils.io import ensure_dir, load_json, save_json  # noqa: E402
from bach_mil.utils.photo_agg import PatchAggConfig, infer_image_patch_agg_backend  # noqa: E402


def _agg_probs(tile_probs: torch.Tensor, agg: str, topk: int) -> torch.Tensor:
    if agg == 'mean_prob':
        return tile_probs.mean(dim=0)
    if agg == 'max_prob':
        return tile_probs.max(dim=0).values
    if agg == 'topk_mean_prob':
        k = int(max(1, min(int(topk), int(tile_probs.shape[0]))))
        return torch.topk(tile_probs, k=k, dim=0).values.mean(dim=0)
    raise ValueError(f'unknown agg={agg}')


def _predict_slide_labels(
    *,
    classes: list[str],
    slide_probs: list[float],
    thresholds: dict | None,
    default_threshold: float,
    normal_fallback: bool,
) -> tuple[list[int], list[str]]:
    pred_flags = []
    active = []
    for i, name in enumerate(classes):
        thr = float(default_threshold)
        if thresholds is not None and name in thresholds:
            thr = float(thresholds[name])
        flag = int(float(slide_probs[i]) >= thr)
        pred_flags.append(flag)
        if flag == 1:
            active.append(name)
    if normal_fallback and ('Normal' in classes) and len(active) == 0:
        idx = classes.index('Normal')
        pred_flags[idx] = 1
        active = ['Normal']
    return pred_flags, active


def _save_wsi_feature_blob(
    *,
    out_path: Path,
    features: torch.Tensor,
    probs: torch.Tensor,
    coords: torch.Tensor,
    slide_id: str,
    classes: list[str],
    encoder_ckpt: str,
    model_name: str,
    level: int,
    tile_size: int,
) -> None:
    blob = {
        'features': features.cpu(),
        'tile_probs': probs.cpu(),
        'coords': coords.cpu(),
        'slide_id': str(slide_id),
        'classes': list(classes),
        'encoder_ckpt': str(encoder_ckpt),
        'model_name': str(model_name),
        'levels': [int(level)],
        'tile_sizes': [int(tile_size)],
    }
    torch.save(blob, out_path)


def _resolve_toggle(value: str, default: bool) -> bool:
    if value == 'auto':
        return bool(default)
    return str(value) == 'on'


def _resolve_choice(value: str | None, default: str) -> str:
    if value is None or value == 'auto':
        return str(default)
    return str(value)


def _manifest_cache_path(cache_dir: str | None, slide_path: Path, *, level: int, tile_size: int, step: int, min_tissue: float) -> Path | None:
    if not cache_dir:
        return None
    mt = str(min_tissue).replace('.', 'p')
    name = f'{slide_path.stem}_L{level}_ts{tile_size}_st{step}_mt{mt}.csv'
    return ensure_dir(cache_dir) / name


def run_wsi(args, defaults: dict, artifacts: dict) -> None:
    backend_name = str(args.backend)
    if backend_name == 'auto':
        backend_name = 'pytorch' if args.variant == 'before' else 'auto'

    backend = create_patch_backend(
        backend=backend_name,
        ckpt_path=args.ckpt,
        model_name=args.model_name,
        backbone_pool=args.backbone_pool,
        backbone_init_values=args.backbone_init_values,
        default_classes=defaults['labels'],
        input_size=args.tile_size,
        onnx_path=args.onnx_path,
        om_path=args.om_path,
        meta_json=args.meta_json,
        device_id=args.device_id,
        amp_dtype=args.torch_amp_dtype,
        channels_last=args.torch_channels_last,
        jit_compile=args.torch_jit_compile,
        om_execution_mode=args.om_execution_mode,
        om_host_io_mode=args.om_host_io_mode,
        om_output_mode=args.om_output_mode,
    )
    out_dir = Path(args.out_dir)
    feature_dir = ensure_dir(out_dir / 'features')
    manifest_dir = ensure_dir(out_dir / 'manifests')
    pred_dir = ensure_dir(out_dir / 'predictions')
    report_dir = ensure_dir(out_dir / 'reports')
    thresholds = load_json(args.thresholds_json) if args.thresholds_json is not None and Path(args.thresholds_json).exists() else None
    tf = T.Compose([
        T.Resize((args.tile_size, args.tile_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    slide_rows = []
    timing_rows = []
    manifest_cache_hits = 0
    manifest_cache_misses = 0
    start_total = time.time()
    try:
        for slide_path in list_wsi_paths(args.input_dir):
            slide_start = time.time()
            manifest_source = 'scan'
            cache_path = _manifest_cache_path(
                args.manifest_cache_dir,
                slide_path,
                level=args.level,
                tile_size=args.tile_size,
                step=args.step,
                min_tissue=args.min_tissue,
            )
            if args.manifest_cache_mode in {'readonly', 'readwrite'} and cache_path is not None and cache_path.exists():
                manifest_df = pd.read_csv(cache_path)
                manifest_source = 'cache'
                manifest_cache_hits += 1
            else:
                manifest_df = scan_slide_for_tiles(
                    slide_path,
                    level=args.level,
                    tile_size=args.tile_size,
                    step=args.step,
                    min_tissue=args.min_tissue,
                )
                manifest_cache_misses += 1
                if args.manifest_cache_mode == 'readwrite' and cache_path is not None:
                    manifest_df.to_csv(cache_path, index=False)
                    manifest_source = 'scan_write_cache'
            manifest_df.to_csv(manifest_dir / f'{slide_path.stem}_manifest.csv', index=False)
            ds = WSITileDataset(manifest_df, transform=tf, return_label=False)
            loader_kwargs = {
                'batch_size': args.batch_size,
                'shuffle': False,
                'num_workers': args.num_workers,
                'pin_memory': args.pin_memory,
            }
            if args.num_workers > 0:
                loader_kwargs['persistent_workers'] = args.persistent_workers
                loader_kwargs['prefetch_factor'] = max(1, int(args.prefetch_factor))
            loader = DataLoader(ds, **loader_kwargs)
            feats, probs, coords = [], [], []
            for batch in tqdm(loader, desc=f'{args.variant}-{slide_path.stem}', leave=False):
                out = backend.predict_batch(batch['image'])
                feats.append(torch.from_numpy(out.features))
                probs.append(torch.from_numpy(out.probs))
                coords.append(torch.stack([batch['x'], batch['y']], dim=1))
            feat_tensor = torch.cat(feats, dim=0) if feats else torch.zeros((0, backend.feature_dim), dtype=torch.float32)
            prob_tensor = torch.cat(probs, dim=0) if probs else torch.zeros((0, len(backend.classes)), dtype=torch.float32)
            coord_tensor = torch.cat(coords, dim=0) if coords else torch.zeros((0, 2), dtype=torch.int64)
            _save_wsi_feature_blob(
                out_path=feature_dir / f'{slide_path.stem}.pt',
                features=feat_tensor,
                probs=prob_tensor,
                coords=coord_tensor,
                slide_id=slide_path.stem,
                classes=backend.classes,
                encoder_ckpt=args.ckpt,
                model_name=args.model_name,
                level=args.level,
                tile_size=args.tile_size,
            )
            slide_prob = _agg_probs(prob_tensor.float(), agg=args.agg, topk=args.topk).cpu().tolist() if len(prob_tensor) else [0.0] * len(backend.classes)
            pred_flags, active = _predict_slide_labels(
                classes=backend.classes,
                slide_probs=slide_prob,
                thresholds=thresholds,
                default_threshold=args.default_threshold,
                normal_fallback=args.normal_fallback,
            )
            row = {
                'slide_id': str(slide_path.stem),
                'pred_labels': ';'.join(active),
                'n_tiles': int(len(manifest_df)),
            }
            for idx, name in enumerate(backend.classes):
                row[f'prob_{name}'] = float(slide_prob[idx])
                row[f'pred_{name}'] = int(pred_flags[idx])
            slide_rows.append(row)
            elapsed = time.time() - slide_start
            timing_rows.append({
                'slide_id': str(slide_path.stem),
                'elapsed_sec': float(elapsed),
                'n_tiles': int(len(manifest_df)),
                'tiles_per_sec': float(len(manifest_df) / max(elapsed, 1e-12)),
                'manifest_source': manifest_source,
            })
    finally:
        backend.close()

    pred_csv = pred_dir / 'slide_predictions.csv'
    timing_csv = report_dir / 'per_slide_timing.csv'
    pd.DataFrame(slide_rows).to_csv(pred_csv, index=False)
    pd.DataFrame(timing_rows).to_csv(timing_csv, index=False)
    total_elapsed = time.time() - start_total
    total_tiles = int(sum(row['n_tiles'] for row in timing_rows))
    num_slides = int(len(timing_rows))
    summary = {
        'task': 'wsi',
        'variant': str(args.variant),
        'backend_requested': str(args.backend),
        'backend_actual': getattr(backend, 'backend_name', str(args.backend)),
        'input_dir': str(args.input_dir),
        'out_dir': str(out_dir),
        'ckpt': str(args.ckpt),
        'onnx_path': str(args.onnx_path) if args.onnx_path is not None else None,
        'om_path': str(args.om_path) if args.om_path is not None else None,
        'meta_json': str(args.meta_json) if args.meta_json is not None else None,
        'thresholds_json': str(args.thresholds_json) if args.thresholds_json is not None else None,
        'model_name': str(args.model_name),
        'level': int(args.level),
        'tile_size': int(args.tile_size),
        'step': int(args.step),
        'min_tissue': float(args.min_tissue),
        'agg': str(args.agg),
        'topk': int(args.topk),
        'total_elapsed_sec': float(total_elapsed),
        'avg_elapsed_sec_per_slide': float(total_elapsed / max(num_slides, 1)),
        'wsi_per_sec': float(num_slides / max(total_elapsed, 1e-12)),
        'total_tiles': total_tiles,
        'tiles_per_sec': float(total_tiles / max(total_elapsed, 1e-12)),
        'avg_tiles_per_slide': float(total_tiles / max(num_slides, 1)),
        'num_slides': num_slides,
        'runtime_profile': str(args.runtime_profile_name),
        'pin_memory': bool(args.pin_memory),
        'persistent_workers': bool(args.persistent_workers),
        'prefetch_factor': int(args.prefetch_factor),
        'manifest_cache_mode': str(args.manifest_cache_mode),
        'manifest_cache_dir': str(args.manifest_cache_dir) if args.manifest_cache_dir is not None else None,
        'manifest_cache_hits': int(manifest_cache_hits),
        'manifest_cache_misses': int(manifest_cache_misses),
        'torch_amp_dtype': str(getattr(backend, 'amp_dtype_name', args.torch_amp_dtype)),
        'torch_jit_compile': bool(getattr(backend, 'jit_compile', args.torch_jit_compile)),
        'torch_channels_last': bool(getattr(backend, 'channels_last', args.torch_channels_last)),
        'torch_channels_last_requested': bool(args.torch_channels_last),
        'om_execution_mode': str(getattr(backend, 'execution_mode', args.om_execution_mode)),
        'om_host_io_mode': str(getattr(backend, 'host_io_mode', args.om_host_io_mode)),
        'om_output_mode': str(getattr(backend, 'output_mode', args.om_output_mode)),
        'feature_dir': str(feature_dir),
        'manifest_dir': str(manifest_dir),
        'pred_csv': str(pred_csv),
        'timing_csv': str(timing_csv),
    }
    save_json(summary, report_dir / 'run_summary.json')


def run_photos(args, defaults: dict, artifacts: dict) -> None:
    backend_name = str(args.backend)
    if backend_name == 'auto':
        backend_name = 'pytorch' if args.variant == 'before' else 'auto'

    backend = create_patch_backend(
        backend=backend_name,
        ckpt_path=args.ckpt,
        model_name=args.model_name,
        backbone_pool=args.backbone_pool,
        backbone_init_values=args.backbone_init_values,
        default_classes=PHOTO_LABELS,
        input_size=args.input_size,
        onnx_path=args.onnx_path,
        om_path=args.om_path,
        meta_json=args.meta_json,
        device_id=args.device_id,
        amp_dtype=args.torch_amp_dtype,
        channels_last=args.torch_channels_last,
        jit_compile=args.torch_jit_compile,
        om_execution_mode=args.om_execution_mode,
        om_host_io_mode=args.om_host_io_mode,
        om_output_mode=args.om_output_mode,
    )
    out_dir = Path(args.out_dir)
    feature_dir = ensure_dir(out_dir / 'features')
    pred_dir = ensure_dir(out_dir / 'predictions')
    report_dir = ensure_dir(out_dir / 'reports')
    image_paths = sorted(list(Path(args.input_dir).glob('*.tif')) + list(Path(args.input_dir).glob('*.png')) + list(Path(args.input_dir).glob('*.jpg')))
    rows = []
    timing_rows = []
    total_crops = 0
    start_total = time.time()
    try:
        for path in tqdm(image_paths, desc=f'{args.variant}-photos'):
            t0 = time.time()
            img = Image.open(path).convert('RGB')
            if args.mode == 'patch_agg':
                cfg = PatchAggConfig(
                    input_size=args.input_size,
                    crop_sizes=tuple(int(x) for x in args.patch_crop_sizes),
                    stride=int(args.patch_stride),
                    topk_per_size=int(args.patch_topk_per_size),
                    min_tissue=float(args.patch_min_tissue),
                    working_max_side=int(args.patch_working_max_side),
                    agg=str(args.patch_agg),
                    logit_topk=int(args.patch_logit_topk),
                )
                prob, meta = infer_image_patch_agg_backend(
                    backend=backend,
                    img=img,
                    num_classes=len(PHOTO_LABELS),
                    cfg=cfg,
                    batch_size=args.batch_size,
                    return_crop_outputs=args.save_features,
                )
                if args.save_features:
                    torch.save(
                        {
                            'image_id': path.name,
                            'features': torch.from_numpy(meta.pop('crop_features')),
                            'logits': torch.from_numpy(meta.pop('crop_logits')),
                            'classes': list(PHOTO_LABELS),
                            'agg_prob': torch.from_numpy(prob),
                            'meta': meta,
                        },
                        feature_dir / f'{path.stem}.pt',
                    )
            else:
                tf = T.Compose([
                    T.Resize((args.input_size, args.input_size)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                x = tf(img).unsqueeze(0)
                out = backend.predict_batch(x)
                prob = softmax_numpy(out.logits)[0]
                meta = {'num_crops': 1}
                if args.save_features:
                    torch.save(
                        {
                            'image_id': path.name,
                            'features': torch.from_numpy(out.features),
                            'logits': torch.from_numpy(out.logits),
                            'classes': list(PHOTO_LABELS),
                            'agg_prob': torch.from_numpy(prob),
                            'meta': meta,
                        },
                        feature_dir / f'{path.stem}.pt',
                    )
            row = {'image_id': path.name, 'pred_label': PHOTO_LABELS[int(prob.argmax())]}
            for idx, name in enumerate(PHOTO_LABELS):
                row[f'prob_{name}'] = float(prob[idx])
            row['num_crops'] = int(meta.get('num_crops', 0))
            total_crops += int(meta.get('num_crops', 0))
            rows.append(row)
            timing_rows.append({'image_id': path.name, 'elapsed_sec': float(time.time() - t0)})
    finally:
        backend.close()

    pred_csv = pred_dir / 'photo_predictions.csv'
    timing_csv = report_dir / 'per_image_timing.csv'
    pd.DataFrame(rows).to_csv(pred_csv, index=False)
    pd.DataFrame(timing_rows).to_csv(timing_csv, index=False)
    total_elapsed = time.time() - start_total
    summary = {
        'task': 'photos',
        'variant': str(args.variant),
        'backend_requested': str(args.backend),
        'backend_actual': getattr(backend, 'backend_name', str(args.backend)),
        'input_dir': str(args.input_dir),
        'out_dir': str(out_dir),
        'ckpt': str(args.ckpt),
        'onnx_path': str(args.onnx_path) if args.onnx_path is not None else None,
        'om_path': str(args.om_path) if args.om_path is not None else None,
        'meta_json': str(args.meta_json) if args.meta_json is not None else None,
        'mode': str(args.mode),
        'total_elapsed_sec': float(total_elapsed),
        'avg_elapsed_sec_per_image': float(total_elapsed / max(len(rows), 1)),
        'images_per_sec': float(len(rows) / max(total_elapsed, 1e-12)),
        'total_crops': int(total_crops),
        'crops_per_sec': float(total_crops / max(total_elapsed, 1e-12)),
        'num_images': int(len(rows)),
        'runtime_profile': str(args.runtime_profile_name),
        'pin_memory': bool(args.pin_memory),
        'persistent_workers': bool(args.persistent_workers),
        'prefetch_factor': int(args.prefetch_factor),
        'torch_amp_dtype': str(getattr(backend, 'amp_dtype_name', args.torch_amp_dtype)),
        'torch_jit_compile': bool(getattr(backend, 'jit_compile', args.torch_jit_compile)),
        'torch_channels_last': bool(getattr(backend, 'channels_last', args.torch_channels_last)),
        'torch_channels_last_requested': bool(args.torch_channels_last),
        'om_execution_mode': str(getattr(backend, 'execution_mode', args.om_execution_mode)),
        'om_host_io_mode': str(getattr(backend, 'host_io_mode', args.om_host_io_mode)),
        'om_output_mode': str(getattr(backend, 'output_mode', args.om_output_mode)),
        'pred_csv': str(pred_csv),
        'timing_csv': str(timing_csv),
        'feature_dir': str(feature_dir) if args.save_features else None,
    }
    save_json(summary, report_dir / 'run_summary.json')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['wsi', 'photos'])
    parser.add_argument('--variant', type=str, required=True, choices=['before', 'after'])
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--save_features', action='store_true')
    parser.add_argument('--backend', type=str, default='auto', choices=['pytorch', 'onnx', 'om', 'auto'])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--onnx_path', type=str, default=None)
    parser.add_argument('--om_path', type=str, default=None)
    parser.add_argument('--meta_json', type=str, default=None)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', type=str, default='auto', choices=['auto', 'on', 'off'])
    parser.add_argument('--persistent_workers', type=str, default='auto', choices=['auto', 'on', 'off'])
    parser.add_argument('--prefetch_factor', type=int, default=None)
    parser.add_argument('--manifest_cache_mode', type=str, default='auto', choices=['auto', 'off', 'readonly', 'readwrite'])
    parser.add_argument('--manifest_cache_dir', type=str, default=None)
    parser.add_argument('--torch_amp_dtype', type=str, default='auto', choices=['auto', 'none', 'fp16', 'bf16'])
    parser.add_argument('--torch_jit_compile', type=str, default='auto', choices=['auto', 'on', 'off'])
    parser.add_argument('--torch_channels_last', type=str, default='auto', choices=['auto', 'on', 'off'])
    parser.add_argument('--om_execution_mode', type=str, default='auto', choices=['auto', 'sync', 'async'])
    parser.add_argument('--om_host_io_mode', type=str, default='auto', choices=['auto', 'legacy', 'buffer_reuse'])
    parser.add_argument('--om_output_mode', type=str, default='auto', choices=['auto', 'both', 'features_only', 'logits_only'])
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--backbone_pool', type=str, default=None)
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--level', type=int, default=None)
    parser.add_argument('--tile_size', type=int, default=None)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--min_tissue', type=float, default=None)
    parser.add_argument('--agg', type=str, default=None)
    parser.add_argument('--topk', type=int, default=None)
    parser.add_argument('--thresholds_json', type=str, default=None)
    parser.add_argument('--default_threshold', type=float, default=1.0)
    parser.add_argument('--normal_fallback', action='store_true')
    parser.add_argument('--mode', type=str, default=None, choices=['resize', 'patch_agg'])
    parser.add_argument('--patch_crop_sizes', type=int, nargs='+', default=None)
    parser.add_argument('--patch_stride', type=int, default=None)
    parser.add_argument('--patch_topk_per_size', type=int, default=None)
    parser.add_argument('--patch_min_tissue', type=float, default=None)
    parser.add_argument('--patch_working_max_side', type=int, default=None)
    parser.add_argument('--patch_agg', type=str, default=None)
    parser.add_argument('--patch_logit_topk', type=int, default=None)
    args = parser.parse_args()

    defaults = get_task_defaults(args.task)
    runtime_profile = get_variant_runtime_profile(args.task, args.variant)
    artifacts = get_default_backend_artifacts(args.task)
    args.runtime_profile_name = 'before_baseline' if args.variant == 'before' else 'after_optimized'
    args.ckpt = args.ckpt or str(defaults['ckpt'])
    args.onnx_path = args.onnx_path or str(artifacts['onnx'])
    args.om_path = args.om_path or str(artifacts['om'])
    if args.meta_json is None:
        default_meta_json = artifacts['meta_json']
        if args.backend == 'om' or (args.backend == 'auto' and Path(args.om_path).exists()):
            default_meta_json = artifacts.get('om_meta_json', default_meta_json)
        args.meta_json = str(default_meta_json)
    args.model_name = args.model_name or defaults['model_name']
    args.backbone_pool = args.backbone_pool or defaults['backbone_pool']
    if args.backbone_init_values is None:
        args.backbone_init_values = defaults['backbone_init_values']
    if args.input_size is None:
        args.input_size = defaults['input_size']
    if args.batch_size is None:
        args.batch_size = defaults['batch_size']
    args.num_workers = runtime_profile['num_workers'] if args.num_workers is None else args.num_workers
    args.pin_memory = _resolve_toggle(args.pin_memory, runtime_profile['pin_memory'])
    args.persistent_workers = _resolve_toggle(args.persistent_workers, runtime_profile['persistent_workers'])
    args.prefetch_factor = runtime_profile['prefetch_factor'] if args.prefetch_factor is None else args.prefetch_factor
    args.manifest_cache_mode = _resolve_choice(args.manifest_cache_mode, runtime_profile['manifest_cache_mode'])
    if args.manifest_cache_dir is None and runtime_profile['manifest_cache_dir'] is not None:
        args.manifest_cache_dir = str(runtime_profile['manifest_cache_dir'])
    args.torch_amp_dtype = _resolve_choice(args.torch_amp_dtype, runtime_profile['torch_amp_dtype'])
    args.torch_jit_compile = _resolve_toggle(args.torch_jit_compile, runtime_profile['torch_jit_compile'])
    args.torch_channels_last = _resolve_toggle(args.torch_channels_last, runtime_profile['torch_channels_last'])
    args.om_execution_mode = _resolve_choice(args.om_execution_mode, runtime_profile['om_execution_mode'])
    args.om_host_io_mode = _resolve_choice(args.om_host_io_mode, runtime_profile['om_host_io_mode'])
    args.om_output_mode = _resolve_choice(args.om_output_mode, runtime_profile['om_output_mode'])

    if args.task == 'wsi':
        args.level = defaults['level'] if args.level is None else args.level
        args.tile_size = defaults['tile_size'] if args.tile_size is None else args.tile_size
        args.step = defaults['step'] if args.step is None else args.step
        args.min_tissue = defaults['min_tissue'] if args.min_tissue is None else args.min_tissue
        args.agg = defaults['agg'] if args.agg is None else args.agg
        args.topk = defaults['topk'] if args.topk is None else args.topk
        args.thresholds_json = args.thresholds_json or str(defaults['thresholds_json'])
        if not args.normal_fallback:
            args.normal_fallback = bool(defaults['normal_fallback'])
        run_wsi(args, defaults, artifacts)
    else:
        args.mode = args.mode or defaults['mode']
        args.patch_crop_sizes = args.patch_crop_sizes or defaults['patch_crop_sizes']
        args.patch_stride = defaults['patch_stride'] if args.patch_stride is None else args.patch_stride
        args.patch_topk_per_size = defaults['patch_topk_per_size'] if args.patch_topk_per_size is None else args.patch_topk_per_size
        args.patch_min_tissue = defaults['patch_min_tissue'] if args.patch_min_tissue is None else args.patch_min_tissue
        args.patch_working_max_side = defaults['patch_working_max_side'] if args.patch_working_max_side is None else args.patch_working_max_side
        args.patch_agg = args.patch_agg or defaults['patch_agg']
        args.patch_logit_topk = defaults['patch_logit_topk'] if args.patch_logit_topk is None else args.patch_logit_topk
        run_photos(args, defaults, artifacts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
