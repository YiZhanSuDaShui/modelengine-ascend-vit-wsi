from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape


CANVAS_W = 1600
CANVAS_H = 920


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {row["config"]: row for row in reader}


class SvgCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 1,
        rx: float = 18,
    ) -> None:
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        stroke_width: float = 1,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"{dash_attr}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        size: int = 24,
        fill: str = "#1f2937",
        weight: str = "400",
        anchor: str = "start",
        family: str = "Microsoft YaHei, Segoe UI, sans-serif",
    ) -> None:
        self.add(
            f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'font-family="{family}" text-anchor="{anchor}">{escape(content)}</text>'
        )

    def save(self, path: Path) -> None:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">'
            + "".join(self.parts)
            + "</svg>"
        )
        path.write_text(svg, encoding="utf-8")


def draw_panel(canvas: SvgCanvas, x: float, y: float, w: float, h: float, title: str, subtitle: str) -> None:
    canvas.rect(x, y, w, h, fill="#ffffff", stroke="#d7dee8", stroke_width=1.4, rx=22)
    canvas.text(x + 24, y + 42, title, size=26, fill="#111827", weight="700")
    canvas.text(x + 24, y + 74, subtitle, size=17, fill="#64748b")
    canvas.line(x + 24, y + 92, x + w - 24, y + 92, stroke="#e5e7eb", stroke_width=1.1)


def draw_bar(
    canvas: SvgCanvas,
    x: float,
    y: float,
    label: str,
    value: float,
    max_value: float,
    color: str,
    value_text: str,
    *,
    track_w: float = 230,
    muted: bool = False,
) -> None:
    canvas.text(x, y, label, size=18, fill="#374151", weight="600")
    canvas.rect(x + 82, y - 16, track_w, 22, fill="#eef2f7", rx=11)
    bar_w = track_w * (value / max_value)
    canvas.rect(x + 82, y - 16, bar_w, 22, fill=color, rx=11)
    if muted:
        canvas.rect(x + 82, y - 16, bar_w, 22, fill="#ffffff", stroke="none", rx=11)
        canvas.rect(x + 82, y - 16, bar_w, 22, fill=color, rx=11)
    canvas.text(x + 82 + track_w + 14, y, value_text, size=18, fill="#111827", weight="700")


def draw_metric_box(
    canvas: SvgCanvas,
    x: float,
    y: float,
    w: float,
    h: float,
    big: str,
    small: str,
    *,
    fill: str,
    color: str,
) -> None:
    canvas.rect(x, y, w, h, fill=fill, rx=18)
    canvas.text(x + w / 2, y + 36, big, size=26, fill=color, weight="800", anchor="middle")
    canvas.text(x + w / 2, y + 64, small, size=15, fill=color, weight="700", anchor="middle")


def build_figure(output_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]

    manifest_summary = read_manifest_summary(repo_root / "logs/A3_output/C_phase/manifest_qc_summary.csv")
    before = read_json(
        repo_root
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/01_before_plain_pytorch/run/reports/run_summary.json"
    )
    origin = read_json(
        repo_root
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/04_after_om_acl_sync/run/reports/run_summary.json"
    )
    mixed = read_json(
        repo_root
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/run/reports/run_summary.json"
    )
    thresholds = read_json(repo_root / "logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json")

    mt20_tiles = int(manifest_summary["mt20"]["tiles_total"])
    mt40_tiles = int(manifest_summary["mt40"]["tiles_total"])
    mt60_tiles = int(manifest_summary["mt60"]["tiles_total"])
    max_tiles = max(mt20_tiles, mt40_tiles, mt60_tiles)

    speedup = before["total_elapsed_sec"] / mixed["total_elapsed_sec"]
    time_drop_pct = (1.0 - mixed["total_elapsed_sec"] / before["total_elapsed_sec"]) * 100.0
    rel_origin_gain = (1.0 - mixed["total_elapsed_sec"] / origin["total_elapsed_sec"]) * 100.0

    canvas = SvgCanvas(CANVAS_W, CANVAS_H)
    canvas.rect(0, 0, CANVAS_W, CANVAS_H, fill="#ffffff", rx=0)

    canvas.text(58, 64, "A3 阶段：WSI 任务级计算裁剪与聚合加速", size=34, fill="#0f172a", weight="800")
    canvas.text(58, 98, "用一张简洁图说明三件事：减少需要计算的 Tile、保持低代价聚合、在统一入口下比较 before/after 实测速度。", size=18, fill="#64748b")

    canvas.rect(58, 118, 1484, 44, fill="#f8fafc", stroke="#e2e8f0", stroke_width=1.1, rx=14)
    canvas.text(
        80,
        146,
        "公平性约束：before/after 共享相同测试集、切块参数、阈值文件、TileAgg topk16 与输出 schema，唯一区别是编码器执行后端。",
        size=17,
        fill="#334155",
        weight="600",
    )

    left_x, mid_x, right_x = 58, 562, 1012
    panel_y = 190

    draw_panel(canvas, left_x, panel_y, 470, 640, "1. 有效 Tile 数控制", "A3 的第一目标是直接压缩 slide 级工作量 M。")
    draw_panel(canvas, mid_x, panel_y, 416, 640, "2. 轻量聚合策略", "A3 的第二目标是在不改标签口径下把 T_agg 压到很小。")
    draw_panel(canvas, right_x, panel_y, 530, 640, "3. 官方 10 张 WSI 实测结果", "统一入口下，任务级裁剪与后端优化产生叠加收益。")

    canvas.rect(left_x + 24, 304, 422, 110, fill="#f8fafc", stroke="#e5e7eb", stroke_width=1.0, rx=16)
    canvas.text(left_x + 42, 338, "正式主线配置", size=22, fill="#111827", weight="700")
    canvas.text(left_x + 42, 370, "level = 1", size=18, fill="#374151")
    canvas.text(left_x + 168, 370, "tile_size = 224", size=18, fill="#374151")
    canvas.text(left_x + 322, 370, "step = 448", size=18, fill="#374151")
    canvas.text(left_x + 42, 398, "min_tissue = 0.4", size=18, fill="#374151")
    canvas.text(left_x + 220, 398, "manifest cache = on", size=18, fill="#374151")

    canvas.text(left_x + 24, 454, "不同组织阈值下的有效 Tile 数", size=20, fill="#111827", weight="700")
    draw_bar(canvas, left_x + 24, 500, "mt20", mt20_tiles, max_tiles, "#94a3b8", f"{mt20_tiles}")
    draw_bar(canvas, left_x + 24, 548, "mt40", mt40_tiles, max_tiles, "#2563eb", f"{mt40_tiles}")
    draw_bar(canvas, left_x + 24, 596, "mt60", mt60_tiles, max_tiles, "#f59e0b", f"{mt60_tiles}")

    canvas.rect(left_x + 24, 642, 422, 126, fill="#eff6ff", stroke="#bfdbfe", stroke_width=1.0, rx=16)
    canvas.text(left_x + 42, 676, "选择 mt40 的原因", size=22, fill="#1d4ed8", weight="700")
    canvas.text(left_x + 42, 708, f"相较 mt20：总 Tile 数减少 {(1 - mt40_tiles / mt20_tiles) * 100:.1f}%", size=18, fill="#1e3a8a")
    canvas.text(left_x + 42, 736, "相较 mt60：保留更多候选病灶区域，不会过度裁剪", size=18, fill="#1e3a8a")
    canvas.text(left_x + 42, 764, "结论：正式主线在样本量与背景抑制之间取中间解", size=18, fill="#1e3a8a")

    canvas.rect(mid_x + 24, 304, 368, 120, fill="#f8fafc", stroke="#e5e7eb", stroke_width=1.0, rx=16)
    canvas.text(mid_x + 42, 338, "任务级耗时近似", size=22, fill="#111827", weight="700")
    canvas.text(mid_x + 42, 374, "T_slide ≈ T_scan + M(T_read + T_pre + T_infer) + T_agg", size=18, fill="#111827")
    canvas.text(mid_x + 42, 404, "当 T_agg 很小时，A3 的优化重心自然落在 M。", size=18, fill="#475569")

    canvas.rect(mid_x + 24, 450, 368, 146, fill="#fffbeb", stroke="#fde68a", stroke_width=1.0, rx=16)
    canvas.text(mid_x + 42, 484, "TileAgg topk16", size=22, fill="#92400e", weight="700")
    canvas.text(mid_x + 42, 518, "P_slide = mean(TopK16 概率)", size=18, fill="#78350f")
    canvas.text(mid_x + 42, 546, "y_hat = I(P_slide ≥ τ)", size=18, fill="#78350f")
    canvas.text(mid_x + 42, 574, "每张 slide 平均 229.0 个 tile，但聚合只看高响应子集", size=18, fill="#78350f")

    canvas.rect(mid_x + 24, 622, 368, 146, fill="#f8fafc", stroke="#e5e7eb", stroke_width=1.0, rx=16)
    canvas.text(mid_x + 42, 656, "正式阈值", size=22, fill="#111827", weight="700")
    canvas.text(mid_x + 42, 690, f"Benign = {thresholds['Benign']:.1f}", size=18, fill="#374151")
    canvas.text(mid_x + 192, 690, f"InSitu = {thresholds['InSitu']:.1f}", size=18, fill="#374151")
    canvas.text(mid_x + 42, 718, f"Invasive = {thresholds['Invasive']:.1f}", size=18, fill="#374151")
    canvas.text(mid_x + 42, 746, "若三类均未激活，则回退为 Normal", size=18, fill="#374151")

    max_elapsed = float(before["total_elapsed_sec"])
    canvas.text(right_x + 24, 314, "总耗时（越短越好）", size=20, fill="#111827", weight="700")
    draw_bar(
        canvas,
        right_x + 24,
        358,
        "before PyTorch",
        float(before["total_elapsed_sec"]),
        max_elapsed,
        "#94a3b8",
        f"{before['total_elapsed_sec']:.2f}s",
        track_w=250,
    )
    canvas.text(right_x + 106, 388, f"平均 {before['avg_elapsed_sec_per_slide']:.2f}s/slide | {before['tiles_per_sec']:.2f} tiles/s", size=17, fill="#64748b")

    draw_bar(
        canvas,
        right_x + 24,
        440,
        "after origin OM",
        float(origin["total_elapsed_sec"]),
        max_elapsed,
        "#f59e0b",
        f"{origin['total_elapsed_sec']:.2f}s",
        track_w=250,
    )
    canvas.text(right_x + 106, 470, f"平均 {origin['avg_elapsed_sec_per_slide']:.2f}s/slide | {origin['tiles_per_sec']:.2f} tiles/s", size=17, fill="#64748b")

    draw_bar(
        canvas,
        right_x + 24,
        522,
        "after mixed OM",
        float(mixed["total_elapsed_sec"]),
        max_elapsed,
        "#0f766e",
        f"{mixed['total_elapsed_sec']:.2f}s",
        track_w=250,
    )
    canvas.text(right_x + 106, 552, f"平均 {mixed['avg_elapsed_sec_per_slide']:.2f}s/slide | {mixed['tiles_per_sec']:.2f} tiles/s", size=17, fill="#64748b")

    canvas.rect(right_x + 24, 600, 482, 78, fill="#f8fafc", stroke="#e5e7eb", stroke_width=1.0, rx=16)
    canvas.text(right_x + 42, 634, "解释方式", size=22, fill="#111827", weight="700")
    canvas.text(right_x + 42, 664, "A3 负责减少“需要算多少 Tile”，A4 mixed OM 负责减少“每个 Tile 算多快”。", size=18, fill="#374151")

    draw_metric_box(canvas, right_x + 24, 704, 148, 82, f"{speedup:.2f}x", "相对 before 加速比", fill="#dcfce7", color="#166534")
    draw_metric_box(canvas, right_x + 182, 704, 148, 82, f"{time_drop_pct:.2f}%", "总耗时下降", fill="#dbeafe", color="#1d4ed8")
    draw_metric_box(canvas, right_x + 340, 704, 166, 82, f"{rel_origin_gain:.2f}%", "相对 origin 再提速", fill="#fef3c7", color="#92400e")

    canvas.text(
        58,
        878,
        "图注建议：A3 阶段通过组织区域筛选、稀疏切块与 TileAgg topk16 聚合，先在任务级减少有效 Tile 数 M，再在统一入口下与 OM 后端优化叠加，最终将官方 10 张测试 WSI 的总耗时从 157.73s 降至 25.26s。",
        size=17,
        fill="#475569",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    output_path = repo_root / "ICT/1.4.4图_A3任务级裁剪聚合加速.svg"
    build_figure(output_path)
    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()
