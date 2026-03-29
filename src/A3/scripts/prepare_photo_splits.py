from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.data.photos import scan_photos
from bach_mil.utils.io import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    photos_root = Path(args.data_root) / 'ICIAR2018_BACH_Challenge' / 'Photos'
    df = scan_photos(photos_root)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    df['fold'] = -1
    for fold, (_, va_idx) in enumerate(skf.split(df.image_path.values, df.label.values)):
        df.loc[va_idx, 'fold'] = fold
    out_dir = ensure_dir(args.out_dir)
    df.to_csv(out_dir / 'photos_folds.csv', index=False)
    all_train = sorted(df.image_path.tolist())
    with open(out_dir / 'photos_train_all.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_train))
    test_root = Path(args.data_root) / 'ICIAR2018_BACH_Challenge_TestDataset' / 'Photos'
    test_paths = sorted(str(p) for p in test_root.glob('*.tif'))
    with open(out_dir / 'photos_test_all.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_paths))
    print(f'Saved split CSV to {out_dir / "photos_folds.csv"}')


if __name__ == '__main__':
    main()
