from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import onnx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.utils.io import ensure_dir, save_json  # noqa: E402


PROFILE_RULES = {
    'plain': {
        'black_list': [],
        'keep_optypes': [],
        'keep_name_keywords': [],
    },
    'ln_softmax': {
        'black_list': ['LayerNormalization', 'Softmax'],
        'keep_optypes': ['LayerNormalization', 'Softmax'],
        'keep_name_keywords': [],
    },
    'ln_softmax_head': {
        'black_list': ['LayerNormalization', 'Softmax', 'Gemm'],
        'keep_optypes': ['LayerNormalization', 'Softmax', 'Gemm'],
        'keep_name_keywords': ['/classifier/'],
    },
    'ln_softmax_head_mlp': {
        'black_list': ['LayerNormalization', 'Softmax', 'Gemm'],
        'keep_optypes': ['LayerNormalization', 'Softmax', 'Gemm'],
        'keep_name_keywords': ['/classifier/', '/mlp/act/'],
    },
    'attn_numeric': {
        'black_list': ['LayerNormalization', 'Softmax', 'Div', 'Sqrt', 'Mul'],
        'keep_optypes': ['LayerNormalization', 'Softmax', 'Div', 'Sqrt', 'Mul'],
        'keep_name_keywords': ['/attn/'],
    },
    'attn_softmax_div_sqrt': {
        'black_list': ['Softmax', 'Div', 'Sqrt'],
        'keep_optypes': [],
        'keep_name_keywords': ['/attn/Softmax', '/attn/Div', '/attn/Sqrt'],
    },
    'attn_softmax_div_sqrt_mul': {
        'black_list': ['Softmax', 'Div', 'Sqrt', 'Mul'],
        'keep_optypes': [],
        'keep_name_keywords': ['/attn/Softmax', '/attn/Div', '/attn/Sqrt', '/attn/Mul'],
    },
    'attn_score_path': {
        'black_list': ['Softmax', 'Div', 'Sqrt', 'Mul', 'MatMul'],
        'keep_optypes': [],
        'keep_name_keywords': ['/attn/Softmax', '/attn/Div', '/attn/Sqrt', '/attn/Mul', '/attn/MatMul'],
    },
    'attn_score_path_norm1': {
        'black_list': ['LayerNormalization', 'Softmax', 'Div', 'Sqrt', 'Mul', 'MatMul'],
        'keep_optypes': [],
        'keep_name_keywords': ['/norm1/LayerNormalization', '/attn/Softmax', '/attn/Div', '/attn/Sqrt', '/attn/Mul', '/attn/MatMul'],
    },
    'attn_score_path_norm1_qkv_proj': {
        'black_list': ['LayerNormalization', 'Softmax', 'Div', 'Sqrt', 'Mul', 'MatMul'],
        'keep_optypes': [],
        'keep_name_keywords': [
            '/norm1/LayerNormalization',
            '/attn/Softmax',
            '/attn/Div',
            '/attn/Sqrt',
            '/attn/Mul',
            '/attn/MatMul',
            '/attn/qkv/MatMul',
            '/attn/proj/MatMul',
        ],
    },
    'attn_matmul_norm_softmax': {
        'black_list': ['LayerNormalization', 'Softmax', 'MatMul'],
        'keep_optypes': ['LayerNormalization', 'Softmax', 'MatMul'],
        'keep_name_keywords': ['/attn/'],
    },
    'attn_mlp_head_full': {
        'black_list': ['LayerNormalization', 'Softmax', 'MatMul', 'Div', 'Sqrt', 'Mul', 'Erf', 'Gemm'],
        'keep_optypes': ['LayerNormalization', 'Softmax', 'MatMul', 'Div', 'Sqrt', 'Mul', 'Erf', 'Gemm'],
        'keep_name_keywords': ['/attn/', '/mlp/act/', '/classifier/'],
    },
    'full_block_guard': {
        'black_list': ['LayerNormalization', 'Softmax', 'MatMul', 'Div', 'Sqrt', 'Mul', 'Erf', 'Gemm', 'Conv'],
        'keep_optypes': ['LayerNormalization', 'Softmax', 'MatMul', 'Div', 'Sqrt', 'Mul', 'Erf', 'Gemm', 'Conv'],
        'keep_name_keywords': ['/backbone/blocks/', '/backbone/norm/', '/classifier/', '/backbone/patch_embed/'],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--profile', type=str, default='ln_softmax_head_mlp', choices=sorted(PROFILE_RULES))
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    model = onnx.load(args.onnx_path)
    nodes = list(model.graph.node)
    op_counter = Counter(node.op_type for node in nodes)
    rules = PROFILE_RULES[args.profile]

    node_rows = []
    keep_dtype_entries: list[str] = []
    seen_keep_entries: set[str] = set()
    for node in nodes:
        node_name = str(node.name)
        keep_by_type = node.op_type in rules['keep_optypes']
        keep_by_name = any(keyword in node_name for keyword in rules['keep_name_keywords'])
        row = {
            'node_name': node_name,
            'op_type': str(node.op_type),
            'keep_by_type': int(keep_by_type),
            'keep_by_name': int(keep_by_name),
        }
        node_rows.append(row)
        if (keep_by_type or keep_by_name) and node_name and node_name not in seen_keep_entries:
            keep_dtype_entries.append(node_name)
            seen_keep_entries.add(node_name)

    node_df = pd.DataFrame(node_rows)
    candidate_df = node_df[(node_df.keep_by_type == 1) | (node_df.keep_by_name == 1)].reset_index(drop=True)

    keep_dtype_path = out_dir / f'keep_dtype_{args.profile}.txt'
    keep_dtype_path.write_text('\n'.join(keep_dtype_entries).strip() + ('\n' if keep_dtype_entries else ''), encoding='utf-8')

    mixlist_path = out_dir / f'modify_mixlist_{args.profile}.json'
    save_json(
        {
            'black-list': {
                'to-add': list(rules['black_list']),
                'to-remove': [],
            },
            'white-list': {
                'to-add': [],
                'to-remove': [],
            },
        },
        mixlist_path,
    )

    summary = {
        'onnx_path': str(args.onnx_path),
        'profile': str(args.profile),
        'total_nodes': int(len(nodes)),
        'op_type_counts': dict(sorted(op_counter.items())),
        'black_list': list(rules['black_list']),
        'keep_dtype_entries': keep_dtype_entries,
        'keep_dtype_path': str(keep_dtype_path),
        'modify_mixlist_path': str(mixlist_path),
        'candidate_csv': str(out_dir / f'candidate_nodes_{args.profile}.csv'),
    }
    save_json(summary, out_dir / f'mix_precision_summary_{args.profile}.json')
    node_df.to_csv(out_dir / f'onnx_node_inventory_{args.profile}.csv', index=False)
    candidate_df.to_csv(out_dir / f'candidate_nodes_{args.profile}.csv', index=False)
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
