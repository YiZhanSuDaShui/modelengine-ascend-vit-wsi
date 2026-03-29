from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_cfg', type=str, required=True)
    parser.add_argument('--input_format', type=str, default='RGB888_U8')
    parser.add_argument('--src_image_size_h', type=int, default=224)
    parser.add_argument('--src_image_size_w', type=int, default=224)
    args = parser.parse_args()

    content = f"""aipp_op {{
    aipp_mode: static
    input_format: {args.input_format}
    src_image_size_w: {int(args.src_image_size_w)}
    src_image_size_h: {int(args.src_image_size_h)}
    csc_switch: false
    rbuv_swap_switch: false
    ax_swap_switch: false
    mean_chn_0: 123.675
    mean_chn_1: 116.28
    mean_chn_2: 103.53
    var_reci_chn_0: 0.017124753831663668
    var_reci_chn_1: 0.01750700280112045
    var_reci_chn_2: 0.017429193899782137
}}
"""
    Path(args.out_cfg).write_text(content, encoding='utf-8')
    print({'out_cfg': args.out_cfg})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
