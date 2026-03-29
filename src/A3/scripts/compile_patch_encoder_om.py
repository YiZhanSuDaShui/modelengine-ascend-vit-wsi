from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.runtime.submission_defaults import get_default_backend_artifacts, get_task_defaults  # noqa: E402
from bach_mil.utils.io import ensure_dir, load_json, save_json  # noqa: E402


def _append_arg(cmd: list[str], key: str, value: str | int | None) -> None:
    if value is None:
        return
    value_str = str(value).strip()
    if not value_str:
        return
    cmd.append(f'--{key}={value_str}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi', choices=['wsi', 'photos'])
    parser.add_argument('--onnx_path', type=str, default=None)
    parser.add_argument('--meta_json', type=str, default=None)
    parser.add_argument('--out_om', type=str, default=None)
    parser.add_argument('--report_json', type=str, default=None)
    parser.add_argument('--atc_bin', type=str, default='atc')
    parser.add_argument('--soc_version', type=str, default='Ascend910B3')
    parser.add_argument('--precision_mode_v2', type=str, default='origin')
    parser.add_argument('--precision_mode', type=str, default=None)
    parser.add_argument('--framework', type=int, default=5)
    parser.add_argument('--modify_mixlist', type=str, default=None)
    parser.add_argument('--keep_dtype', type=str, default=None)
    parser.add_argument('--insert_op_conf', type=str, default=None)
    parser.add_argument('--dynamic_batch_size', type=str, default=None)
    parser.add_argument('--op_select_implmode', type=str, default=None)
    parser.add_argument('--optypelist_for_implmode', type=str, default=None)
    parser.add_argument('--op_precision_mode', type=str, default=None)
    args = parser.parse_args()

    defaults = get_task_defaults(args.task)
    artifacts = get_default_backend_artifacts(args.task)
    onnx_path = Path(args.onnx_path) if args.onnx_path is not None else artifacts['onnx']
    meta_json = Path(args.meta_json) if args.meta_json is not None else artifacts['meta_json']
    out_om = Path(args.out_om) if args.out_om is not None else artifacts['om']
    report_json = Path(args.report_json) if args.report_json is not None else artifacts['om_compile_json']
    ensure_dir(out_om.parent)
    ensure_dir(report_json.parent)

    out_prefix = out_om.with_suffix('') if out_om.suffix == '.om' else out_om
    meta = load_json(meta_json)
    input_shape = meta['input_shape']
    input_name = meta.get('input_name', 'input')
    input_shape_arg = f'{input_name}:{",".join(str(int(x)) for x in input_shape)}'

    cmd = [
        args.atc_bin,
        f'--model={onnx_path}',
        f'--framework={int(args.framework)}',
        f'--output={out_prefix}',
        '--input_format=NCHW',
        f'--input_shape={input_shape_arg}',
        f'--soc_version={args.soc_version}',
        f'--precision_mode_v2={args.precision_mode_v2}',
    ]
    _append_arg(cmd, 'precision_mode', args.precision_mode)
    _append_arg(cmd, 'modify_mixlist', args.modify_mixlist)
    _append_arg(cmd, 'keep_dtype', args.keep_dtype)
    _append_arg(cmd, 'insert_op_conf', args.insert_op_conf)
    _append_arg(cmd, 'dynamic_batch_size', args.dynamic_batch_size)
    _append_arg(cmd, 'op_select_implmode', args.op_select_implmode)
    _append_arg(cmd, 'optypelist_for_implmode', args.optypelist_for_implmode)
    _append_arg(cmd, 'op_precision_mode', args.op_precision_mode)

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    actual_om = out_prefix.with_suffix('.om')
    om_meta_json = actual_om.with_suffix('.meta.json')
    if proc.returncode == 0 and meta_json.exists() and meta_json.resolve() != om_meta_json.resolve():
        shutil.copyfile(meta_json, om_meta_json)

    summary = {
        'task': str(args.task),
        'status': 'ok' if proc.returncode == 0 and actual_om.exists() else 'failed',
        'ckpt': str(defaults['ckpt']),
        'onnx_path': str(onnx_path),
        'meta_json': str(meta_json),
        'out_om': str(actual_om),
        'om_meta_json': str(om_meta_json),
        'elapsed_sec': float(elapsed),
        'returncode': int(proc.returncode),
        'command': cmd,
        'framework': int(args.framework),
        'soc_version': str(args.soc_version),
        'precision_mode_v2': str(args.precision_mode_v2),
        'precision_mode': str(args.precision_mode) if args.precision_mode is not None else None,
        'modify_mixlist': str(args.modify_mixlist) if args.modify_mixlist is not None else None,
        'keep_dtype': str(args.keep_dtype) if args.keep_dtype is not None else None,
        'insert_op_conf': str(args.insert_op_conf) if args.insert_op_conf is not None else None,
        'dynamic_batch_size': str(args.dynamic_batch_size) if args.dynamic_batch_size is not None else None,
        'op_select_implmode': str(args.op_select_implmode) if args.op_select_implmode is not None else None,
        'optypelist_for_implmode': str(args.optypelist_for_implmode) if args.optypelist_for_implmode is not None else None,
        'op_precision_mode': str(args.op_precision_mode) if args.op_precision_mode is not None else None,
        'stdout': proc.stdout[-20000:],
        'stderr': proc.stderr[-20000:],
    }
    save_json(summary, report_json)
    print(summary)
    return 0 if summary['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
