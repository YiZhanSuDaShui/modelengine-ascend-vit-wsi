from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

STEP_TO_SCRIPT = {
    'step00': 'run_step00_audit.py',
    'step01': 'run_step01_mixed_precision.py',
    'step02': 'run_step02_acl_runtime.py',
    'step03': 'run_step03_quantization.py',
    'step04': 'run_step04_distill.py',
    'step05': 'run_step05_prune.py',
    'step06': 'run_step06_aipp.py',
    'step07': 'run_step07_final_report.py',
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi')
    parser.add_argument('--steps', nargs='+', default=['step00', 'step01', 'step02', 'step03', 'step04', 'step05', 'step06', 'step07'])
    args = parser.parse_args()

    for step in args.steps:
        if step not in STEP_TO_SCRIPT:
            raise ValueError(f'unknown step={step}')
        cmd = [sys.executable, str(ROOT / 'scripts' / STEP_TO_SCRIPT[step]), '--task', str(args.task)]
        print({'running': cmd})
        proc = subprocess.run(cmd, cwd=str(ROOT.parents[1]))
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
