from __future__ import annotations

from pathlib import Path

DEFAULT_LABELS = ['Normal', 'Benign', 'InSitu', 'Invasive']
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPORT_ROOT = PROJECT_ROOT / 'logs' / 'A3_output' / 'submission_closure'
DEFAULT_OPTIMIZATION_ROOT = DEFAULT_REPORT_ROOT / 'optimization_stepwise'


def _first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def get_task_defaults(task: str) -> dict:
    task = str(task).lower()
    if task == 'wsi':
        return {
            'task': 'wsi',
            'ckpt': PROJECT_ROOT / 'logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt',
            'thresholds_json': PROJECT_ROOT / 'logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json',
            'labels': ['Normal', 'Benign', 'InSitu', 'Invasive'],
            'proxy_labels': ['Benign', 'InSitu', 'Invasive'],
            'model_name': 'vit_large_patch16_224',
            'backbone_pool': 'token',
            'backbone_init_values': 1.0,
            'input_size': 224,
            'batch_size': 64,
            'level': 1,
            'tile_size': 224,
            'step': 448,
            'min_tissue': 0.4,
            'agg': 'topk_mean_prob',
            'topk': 16,
            'normal_fallback': True,
            'bag_csv': PROJECT_ROOT / 'data/BACH/derived/split/wsi_train_bags_mt40.csv',
            'official_test_input': PROJECT_ROOT / 'data/BACH/ICIAR2018_BACH_Challenge_TestDataset/WSI',
        }
    if task == 'photos':
        return {
            'task': 'photos',
            'ckpt': _first_existing([
                PROJECT_ROOT / 'logs/A3_output/B_phase/stage1_uni_large_cv5_ms512_1024_1536_v1/fold0/best.pt',
                PROJECT_ROOT / 'logs/A3_output/B_phase/stage1_final_fold0/best.pt',
            ]),
            'labels': list(DEFAULT_LABELS),
            'model_name': 'vit_large_patch16_224',
            'backbone_pool': 'token',
            'backbone_init_values': 1.0,
            'input_size': 224,
            'batch_size': 64,
            'mode': 'patch_agg',
            'patch_crop_sizes': [512, 1024, 1536],
            'patch_stride': 256,
            'patch_topk_per_size': 8,
            'patch_min_tissue': 0.4,
            'patch_working_max_side': 1536,
            'patch_agg': 'topk_mean_logit',
            'patch_logit_topk': 12,
            'split_csv': PROJECT_ROOT / 'data/BACH/derived/split/photos_folds.csv',
            'official_test_input': PROJECT_ROOT / 'data/BACH/ICIAR2018_BACH_Challenge_TestDataset/Photos',
            'summary_json': PROJECT_ROOT / 'logs/A3_output/reports/stage1_uni_large_cv5_ms512_1024_1536_summary.json',
        }
    raise ValueError(f'unknown task={task}')


def get_default_backend_artifacts(task: str) -> dict:
    task = str(task).lower()
    if task != 'wsi':
        task = 'photos'
    root = DEFAULT_REPORT_ROOT / 'offline_models' / task
    mixed_root = (
        DEFAULT_OPTIMIZATION_ROOT
        / '01_mixed_float16_keep_dtype_modify_mixlist'
        / 'artifacts'
        / 'refined_attn_score_path_norm1'
    )
    om_candidates: list[Path] = []
    om_meta_candidates: list[Path] = []
    om_compile_candidates: list[Path] = []
    om_parity_candidates: list[Path] = []
    if task == 'wsi':
        om_candidates.append(mixed_root / 'wsi_attn_score_path_norm1.om')
        om_meta_candidates.append(mixed_root / 'wsi_attn_score_path_norm1.meta.json')
        om_compile_candidates.append(mixed_root / 'compile_summary.json')
        om_parity_candidates.extend(
            [
                mixed_root / 'parity_real.json',
                mixed_root / 'parity_random.json',
            ]
        )
    om_candidates.extend(
        [
            root / f'{task}_patch_encoder_bs64_origin.om',
            root / f'{task}_patch_encoder_bs64.om',
        ]
    )
    om_meta_candidates.extend(
        [
            root / f'{task}_patch_encoder_bs64_origin.meta.json',
            root / f'{task}_patch_encoder_bs64.meta.json',
        ]
    )
    om_compile_candidates.extend(
        [
            root / f'{task}_om_compile_summary_origin.json',
            root / f'{task}_om_compile_summary.json',
        ]
    )
    om_parity_candidates.extend(
        [
            root / f'{task}_om_parity_summary_origin.json',
            root / f'{task}_om_parity_summary.json',
        ]
    )
    om_path = _first_existing(om_candidates)
    om_meta_json = _first_existing(om_meta_candidates)
    return {
        'root': root,
        'onnx': root / f'{task}_patch_encoder_bs64.onnx',
        'om': om_path,
        'meta_json': root / f'{task}_patch_encoder_bs64.meta.json',
        'om_meta_json': om_meta_json,
        'onnx_export_json': root / f'{task}_onnx_export_summary.json',
        'om_compile_json': _first_existing(om_compile_candidates),
        'onnx_parity_json': root / f'{task}_onnx_parity_summary.json',
        'om_parity_json': _first_existing(om_parity_candidates),
    }


def get_variant_runtime_profile(task: str, variant: str) -> dict:
    task = str(task).lower()
    variant = str(variant).lower()
    if variant not in {'before', 'after'}:
        raise ValueError(f'unknown variant={variant}')

    base = {
        'num_workers': 8,
        'pin_memory': False,
        'persistent_workers': False,
        'prefetch_factor': 1,
        'manifest_cache_mode': 'off',
        'manifest_cache_dir': None,
        'torch_amp_dtype': 'none',
        'torch_jit_compile': False,
        'torch_channels_last': False,
        'om_execution_mode': 'sync',
        'om_host_io_mode': 'legacy',
        'om_output_mode': 'both',
    }

    if variant == 'before':
        return base

    cache_root = DEFAULT_REPORT_ROOT / 'cache' / task
    optimized = dict(base)
    optimized.update(
        {
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
            'manifest_cache_mode': 'readwrite' if task == 'wsi' else 'off',
            'manifest_cache_dir': cache_root if task == 'wsi' else None,
            'torch_amp_dtype': 'fp16',
            'torch_jit_compile': True,
            'torch_channels_last': True,
            'om_execution_mode': 'sync',
            'om_host_io_mode': 'legacy',
            'om_output_mode': 'both',
        }
    )
    return optimized


def get_acceleration_rounds(task: str) -> list[dict]:
    task = str(task).lower()
    defaults = get_task_defaults(task)
    artifacts = get_default_backend_artifacts(task)
    cache_root = DEFAULT_REPORT_ROOT / 'cache' / task

    if task == 'wsi':
        proxy_before_csv = PROJECT_ROOT / 'logs/A3_output/E_phase/tileagg_cv_L1_s448_uniStage1p5_topk16_v1/cv_summary_mean_std.csv'
        proxy_after_csv = _first_existing([
            PROJECT_ROOT / 'logs/A3_output/submission_closure/proxy_eval/wsi_mixed_om_attn_score_path_norm1_cv/cv_summary_mean_std.csv',
            PROJECT_ROOT / 'logs/A3_output/submission_closure/proxy_eval/wsi_after_om_origin_nw8_cv/cv_summary_mean_std.csv',
        ])
        return [
            {
                'name': '01_before_plain_pytorch',
                'task': 'wsi',
                'variant': 'before',
                'backend': 'pytorch',
                'execution_scope': 'full',
                'description': '同 workers 配置下关闭工程优化、关闭离线后端的公平 before 基线。',
                'methods_enabled': ['PyTorch eager', '同 workers', '无流水', '无缓存', '无 OM', '无低精度'],
                'overrides': {
                    'num_workers': 8,
                    'pin_memory': 'off',
                    'persistent_workers': 'off',
                    'prefetch_factor': 1,
                    'manifest_cache_mode': 'off',
                    'torch_amp_dtype': 'none',
                    'torch_jit_compile': 'off',
                    'torch_channels_last': 'off',
                    'om_execution_mode': 'sync',
                },
                'model_refs': {
                    'ckpt': str(defaults['ckpt']),
                    'thresholds_json': str(defaults['thresholds_json']),
                },
                'existing_run_dir': str(PROJECT_ROOT / 'logs/A3_output/submission_closure/official_runs/wsi_before_unified'),
                'proxy_metrics_csv': str(proxy_before_csv) if proxy_before_csv.exists() else None,
            },
            {
                'name': '02_after_pytorch_engineered',
                'task': 'wsi',
                'variant': 'after',
                'backend': 'pytorch',
                'execution_scope': 'full',
                'description': '仅用 PyTorch 主线叠加工程优化，不切换离线模型，用于衡量工程加速贡献。',
                'methods_enabled': ['PyTorch eager', 'AMP(fp16)', 'compile', 'prefetch', 'persistent_workers', 'pin_memory', 'manifest cache'],
                'overrides': {
                    'num_workers': 8,
                    'pin_memory': 'on',
                    'persistent_workers': 'on',
                    'prefetch_factor': 4,
                    'manifest_cache_mode': 'readwrite',
                    'manifest_cache_dir': str(cache_root),
                    'torch_amp_dtype': 'fp16',
                    'torch_jit_compile': 'on',
                    'torch_channels_last': 'on',
                    'om_execution_mode': 'sync',
                },
                'model_refs': {
                    'ckpt': str(defaults['ckpt']),
                    'thresholds_json': str(defaults['thresholds_json']),
                },
                'proxy_metrics_csv': None,
            },
            {
                'name': '03_after_onnx_cpu_smoke',
                'task': 'wsi',
                'variant': 'after',
                'backend': 'onnx',
                'execution_scope': 'smoke',
                'smoke_items': 1,
                'description': '验证 ONNX 导出链路与统一入口可复现；当前环境仅 CPUExecutionProvider，不纳入正式 Ascend 速度主线。',
                'methods_enabled': ['ONNX export', '统一入口 smoke 验证'],
                'overrides': {
                    'num_workers': 8,
                    'pin_memory': 'on',
                    'persistent_workers': 'on',
                    'prefetch_factor': 4,
                    'manifest_cache_mode': 'readwrite',
                    'manifest_cache_dir': str(cache_root),
                    'torch_amp_dtype': 'none',
                    'torch_jit_compile': 'off',
                    'torch_channels_last': 'off',
                    'om_execution_mode': 'sync',
                },
                'model_refs': {
                    'onnx': str(artifacts['onnx']),
                    'meta_json': str(artifacts['meta_json']),
                    'ckpt': str(defaults['ckpt']),
                },
                'proxy_metrics_csv': None,
            },
            {
                'name': '04_after_om_acl_sync',
                'task': 'wsi',
                'variant': 'after',
                'backend': 'om',
                'execution_scope': 'full',
                'description': '当前实测最优正式 after：推荐 mixed OM(attn_score_path_norm1)+ACL sync+工程流水+缓存命中，在统一入口下给出最快正式结果。',
                'methods_enabled': ['OM(mixed attn_score_path_norm1)', 'ACL sync', 'prefetch', 'persistent_workers', 'pin_memory', 'manifest cache'],
                'overrides': {
                    'num_workers': 8,
                    'pin_memory': 'on',
                    'persistent_workers': 'on',
                    'prefetch_factor': 4,
                    'manifest_cache_mode': 'readwrite',
                    'manifest_cache_dir': str(cache_root),
                    'torch_amp_dtype': 'fp16',
                    'torch_jit_compile': 'on',
                    'torch_channels_last': 'on',
                    'om_execution_mode': 'sync',
                },
                'model_refs': {
                    'om': str(artifacts['om']),
                    'om_meta_json': str(artifacts['om_meta_json']),
                    'ckpt': str(defaults['ckpt']),
                    'thresholds_json': str(defaults['thresholds_json']),
                },
                'proxy_metrics_csv': str(proxy_after_csv) if proxy_after_csv.exists() else None,
            },
            {
                'name': '05_after_om_acl_async',
                'task': 'wsi',
                'variant': 'after',
                'backend': 'om',
                'execution_scope': 'full',
                'description': 'ACL async 对照轮：与推荐 mixed OM 同任务同输出，只切换异步执行以验证其收益是否稳定。',
                'methods_enabled': ['OM(mixed attn_score_path_norm1)', 'ACL async', 'prefetch', 'persistent_workers', 'pin_memory', 'manifest cache', '统一入口 async 对照'],
                'overrides': {
                    'num_workers': 8,
                    'pin_memory': 'on',
                    'persistent_workers': 'on',
                    'prefetch_factor': 4,
                    'manifest_cache_mode': 'readwrite',
                    'manifest_cache_dir': str(cache_root),
                    'torch_amp_dtype': 'fp16',
                    'torch_jit_compile': 'on',
                    'torch_channels_last': 'on',
                    'om_execution_mode': 'async',
                },
                'model_refs': {
                    'om': str(artifacts['om']),
                    'om_meta_json': str(artifacts['om_meta_json']),
                    'ckpt': str(defaults['ckpt']),
                    'thresholds_json': str(defaults['thresholds_json']),
                },
                'proxy_metrics_csv': str(proxy_after_csv) if proxy_after_csv.exists() else None,
            },
            {
                'name': '06_after_om_acl_async_cachewarm',
                'task': 'wsi',
                'variant': 'after',
                'backend': 'om',
                'execution_scope': 'full',
                'description': '在推荐 mixed OM after 基础上复用已生成的 manifest cache，衡量重复运行场景的极限性能。',
                'methods_enabled': ['OM(mixed attn_score_path_norm1)', 'ACL async', 'prefetch', 'persistent_workers', 'pin_memory', 'manifest cache warm'],
                'overrides': {
                    'num_workers': 8,
                    'pin_memory': 'on',
                    'persistent_workers': 'on',
                    'prefetch_factor': 4,
                    'manifest_cache_mode': 'readonly',
                    'manifest_cache_dir': str(cache_root),
                    'torch_amp_dtype': 'fp16',
                    'torch_jit_compile': 'on',
                    'torch_channels_last': 'on',
                    'om_execution_mode': 'async',
                },
                'model_refs': {
                    'om': str(artifacts['om']),
                    'om_meta_json': str(artifacts['om_meta_json']),
                    'ckpt': str(defaults['ckpt']),
                    'thresholds_json': str(defaults['thresholds_json']),
                },
                'proxy_metrics_csv': str(proxy_after_csv) if proxy_after_csv.exists() else None,
            },
        ]

    if task == 'photos':
        proxy_after_csv = PROJECT_ROOT / 'logs/A3_output/submission_closure/proxy_eval/photos_after_om_origin/cv_summary_mean_std.csv'
        return [
            {
                'name': '01_before_plain_pytorch',
                'task': 'photos',
                'variant': 'before',
                'backend': 'pytorch',
                'execution_scope': 'full',
                'description': 'Photos before 基线。',
                'methods_enabled': ['PyTorch eager', '无工程优化', '无 OM'],
                'overrides': {
                    'num_workers': 8,
                    'pin_memory': 'off',
                    'persistent_workers': 'off',
                    'prefetch_factor': 1,
                    'torch_amp_dtype': 'none',
                    'torch_jit_compile': 'off',
                    'torch_channels_last': 'off',
                    'om_execution_mode': 'sync',
                },
                'model_refs': {'ckpt': str(defaults['ckpt'])},
                'proxy_metrics_csv': str(defaults['summary_json']) if Path(defaults['summary_json']).exists() else None,
            },
            {
                'name': '02_after_pytorch_engineered',
                'task': 'photos',
                'variant': 'after',
                'backend': 'pytorch',
                'execution_scope': 'full',
                'description': 'Photos PyTorch 工程优化轮。',
                'methods_enabled': ['PyTorch eager', 'AMP(fp16)', 'compile'],
                'overrides': {
                    'num_workers': 8,
                    'pin_memory': 'on',
                    'persistent_workers': 'on',
                    'prefetch_factor': 4,
                    'torch_amp_dtype': 'fp16',
                    'torch_jit_compile': 'on',
                    'torch_channels_last': 'on',
                    'om_execution_mode': 'sync',
                },
                'model_refs': {'ckpt': str(defaults['ckpt'])},
                'proxy_metrics_csv': None,
            },
            {
                'name': '03_after_onnx_cpu',
                'task': 'photos',
                'variant': 'after',
                'backend': 'onnx',
                'execution_scope': 'full',
                'description': 'Photos ONNX 路线；当前环境为 CPUExecutionProvider。',
                'methods_enabled': ['ONNX export'],
                'overrides': {
                    'num_workers': 8,
                    'pin_memory': 'on',
                    'persistent_workers': 'on',
                    'prefetch_factor': 4,
                    'torch_amp_dtype': 'none',
                    'torch_jit_compile': 'off',
                    'torch_channels_last': 'off',
                    'om_execution_mode': 'sync',
                },
                'model_refs': {
                    'onnx': str(artifacts['onnx']),
                    'meta_json': str(artifacts['meta_json']),
                    'ckpt': str(defaults['ckpt']),
                },
                'proxy_metrics_csv': None,
            },
            {
                'name': '04_after_om_acl_sync',
                'task': 'photos',
                'variant': 'after',
                'backend': 'om',
                'execution_scope': 'full',
                'description': 'Photos OM 同步执行轮。',
                'methods_enabled': ['OM(origin)', 'ACL sync'],
                'overrides': {
                    'num_workers': 8,
                    'pin_memory': 'on',
                    'persistent_workers': 'on',
                    'prefetch_factor': 4,
                    'torch_amp_dtype': 'none',
                    'torch_jit_compile': 'off',
                    'torch_channels_last': 'off',
                    'om_execution_mode': 'sync',
                },
                'model_refs': {
                    'om': str(artifacts['om']),
                    'om_meta_json': str(artifacts['om_meta_json']),
                    'ckpt': str(defaults['ckpt']),
                },
                'proxy_metrics_csv': None,
            },
            {
                'name': '05_after_om_acl_async',
                'task': 'photos',
                'variant': 'after',
                'backend': 'om',
                'execution_scope': 'full',
                'description': 'Photos 正式 OM 异步轮。',
                'methods_enabled': ['OM(origin)', 'ACL async'],
                'overrides': {
                    'num_workers': 8,
                    'pin_memory': 'on',
                    'persistent_workers': 'on',
                    'prefetch_factor': 4,
                    'torch_amp_dtype': 'none',
                    'torch_jit_compile': 'off',
                    'torch_channels_last': 'off',
                    'om_execution_mode': 'async',
                },
                'model_refs': {
                    'om': str(artifacts['om']),
                    'om_meta_json': str(artifacts['om_meta_json']),
                    'ckpt': str(defaults['ckpt']),
                },
                'proxy_metrics_csv': str(proxy_after_csv) if proxy_after_csv.exists() else None,
            },
        ]

    raise ValueError(f'unknown task={task}')
