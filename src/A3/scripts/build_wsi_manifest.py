from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.data.wsi_manifest import build_annotated_tile_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--wsi_dir', type=str, required=True)
    parser.add_argument('--out_csv', type=str, required=True)
    parser.add_argument('--out_bag_csv', type=str, required=True)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--tile_size', type=int, default=224)
    parser.add_argument('--step', type=int, default=224)
    parser.add_argument('--min_tissue', type=float, default=0.4)
    parser.add_argument('--label_names', nargs='+', default=['Normal', 'Benign', 'InSitu', 'Invasive'])
    parser.add_argument('--no_normal_from_unannotated', action='store_true')
    args = parser.parse_args()

    tiles_df, bags_df = build_annotated_tile_manifest(
        wsi_dir=args.wsi_dir,
        out_csv=args.out_csv,
        out_bag_csv=args.out_bag_csv,
        level=args.level,
        tile_size=args.tile_size,
        step=args.step,
        min_tissue=args.min_tissue,
        label_names=args.label_names,
        include_normal_from_unannotated=not args.no_normal_from_unannotated,
    )
    print('tiles:', len(tiles_df))
    print('bags :', len(bags_df))
    print(f'saved tile manifest -> {args.out_csv}')
    print(f'saved bag labels   -> {args.out_bag_csv}')


if __name__ == '__main__':
    main()
