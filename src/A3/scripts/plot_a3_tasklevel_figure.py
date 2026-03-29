from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "ICT"

RUN_CONFIGS = {
    "before": {
        "label": "before PyTorch",
        "color": "#4C78A8",
        "summary": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/01_before_plain_pytorch/run/reports/run_summary.json",
        "timing": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/01_before_plain_pytorch/run/reports/per_slide_timing.csv",
    },
    "origin": {
        "label": "after origin OM",
        "color": "#F58518",
        "summary": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/04_after_om_acl_sync/run/reports/run_summary.json",
        "timing": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/04_after_om_acl_sync/run/reports/per_slide_timing.csv",
    },
    "mixed": {
        "label": "after mixed OM",
        "color": "#54A24B",
        "summary": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/run/reports/run_summary.json",
        "timing": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/run/reports/per_slide_timing.csv",
    },
}

THRESHOLDS_JSON = PROJECT_ROOT / "logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json"

SVG_W = 1600
SVG_H = 1080
FONT_STACK = "Microsoft YaHei, SimHei, Arial, sans-serif"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_text(text: str) -> str:
    return escape(text)


class SvgCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.elements: list[str] = []

    def add(self, content: str) -> None:
        self.elements.append(content)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1.0,
        rx: float = 0.0,
        dash: str | None = None,
        opacity: float | None = None,
    ) -> None:
        attrs = [
            f'x="{x:.2f}"',
            f'y="{y:.2f}"',
            f'width="{w:.2f}"',
            f'height="{h:.2f}"',
            f'fill="{fill}"',
            f'stroke="{stroke}"',
            f'stroke-width="{stroke_width:.2f}"',
        ]
        if rx > 0:
            attrs.append(f'rx="{rx:.2f}"')
            attrs.append(f'ry="{rx:.2f}"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity:.3f}"')
        self.add(f"<rect {' '.join(attrs)} />")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = "#000000",
        stroke_width: float = 1.0,
        dash: str | None = None,
        marker_end: bool = False,
        opacity: float | None = None,
    ) -> None:
        attrs = [
            f'x1="{x1:.2f}"',
            f'y1="{y1:.2f}"',
            f'x2="{x2:.2f}"',
            f'y2="{y2:.2f}"',
            f'stroke="{stroke}"',
            f'stroke-width="{stroke_width:.2f}"',
            'fill="none"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if marker_end:
            attrs.append('marker-end="url(#arrow)"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity:.3f}"')
        self.add(f"<line {' '.join(attrs)} />")

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        stroke: str,
        stroke_width: float = 2.0,
        fill: str = "none",
        opacity: float | None = None,
    ) -> None:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        attrs = [
            f'points="{pts}"',
            f'stroke="{stroke}"',
            f'stroke-width="{stroke_width:.2f}"',
            f'fill="{fill}"',
        ]
        if opacity is not None:
            attrs.append(f'opacity="{opacity:.3f}"')
        self.add(f"<polyline {' '.join(attrs)} />")

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 1.0,
        opacity: float | None = None,
    ) -> None:
        attrs = [
            f'cx="{cx:.2f}"',
            f'cy="{cy:.2f}"',
            f'r="{r:.2f}"',
            f'fill="{fill}"',
            f'stroke="{stroke}"',
            f'stroke-width="{stroke_width:.2f}"',
        ]
        if opacity is not None:
            attrs.append(f'opacity="{opacity:.3f}"')
        self.add(f"<circle {' '.join(attrs)} />")

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: int = 14,
        fill: str = "#203040",
        weight: str = "normal",
        anchor: str = "start",
        line_height: float = 1.3,
    ) -> None:
        lines = fmt_text(text).split("\n")
        if len(lines) == 1:
            self.add(
                f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT_STACK}" '
                f'font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{lines[0]}</text>'
            )
            return
        tspans = []
        for idx, line in enumerate(lines):
            dy = 0 if idx == 0 else size * line_height
            tspans.append(f'<tspan x="{x:.2f}" dy="{dy:.2f}">{line}</tspan>')
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT_STACK}" '
            f'font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">'
            + "".join(tspans)
            + "</text>"
        )

    def to_svg(self) -> str:
        defs = f"""
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#6C7A89" />
  </marker>
</defs>
"""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="white" />\n'
            f"{defs}\n"
            + "\n".join(self.elements)
            + "\n</svg>\n"
        )


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    x_mean = mean(xs)
    y_mean = mean(ys)
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = ss_xy / ss_xx if ss_xx else 0.0
    intercept = y_mean - slope * x_mean
    preds = [slope * x + intercept for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return slope, intercept, r2


def map_range(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return dst_min
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def panel_header(svg: SvgCanvas, x: float, y: float, label: str, title: str, width: float) -> None:
    svg.rect(x, y, width, 34, fill="#E2F0D9", stroke="none", rx=8)
    svg.text(x + 16, y + 23, f"{label}  {title}", size=18, weight="bold")


def draw_box(svg: SvgCanvas, x: float, y: float, w: float, h: float, text: str, *, fill: str, stroke: str, size: int = 16) -> None:
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, stroke_width=1.6, rx=16)
    lines = text.split("\n")
    total_h = (len(lines) - 1) * size * 1.3
    start_y = y + h / 2.0 - total_h / 2.0
    svg.text(x + w / 2.0, start_y, text, size=size, anchor="middle")


def draw_candidate_grid(svg: SvgCanvas, x: float, y: float) -> None:
    selected = {0, 2, 4, 5, 9, 11}
    idx = 0
    for row in range(3):
        for col in range(4):
            fill = "#9BD3AE" if idx in selected else "#D8E2EC"
            stroke = "#4C9F70" if idx in selected else "#AAB7C4"
            svg.rect(x + col * 36, y + row * 52, 28, 42, fill=fill, stroke=stroke, stroke_width=1.2, rx=4)
            idx += 1
    svg.text(x + 64, y - 10, "候选 Tile 采样与组织筛选", size=15, anchor="middle")
    svg.text(x + 64, y + 170, "稀疏扫描 + r_m ≥ 0.4 过滤背景", size=13, fill="#506070", anchor="middle")


def draw_topk_bars(svg: SvgCanvas, x: float, y: float) -> None:
    heights = [88, 80, 72, 66, 46, 40, 34, 28]
    for idx, height in enumerate(heights):
        fill = "#4C78A8" if idx < 4 else "#C9D6E3"
        svg.rect(x + idx * 22, y + 100 - height, 14, height, fill=fill, stroke="none", rx=2)
    svg.text(x + 78, y - 12, "TopK=16 per class", size=15, anchor="middle")


def draw_panel_a(svg: SvgCanvas, thresholds: dict[str, float]) -> None:
    x = 40
    y = 90
    w = 1520
    h = 350
    svg.rect(x, y, w, h, fill="#FCFDFE", stroke="#D6E0EA", stroke_width=1.2, rx=18)
    panel_header(svg, x + 16, y + 16, "A", "A3 原理：控制有效 Tile 数 M，并用低代价聚合完成 Slide 判定", 820)

    svg.text(x + 30, y + 88, "T_slide ≈ T_scan + M (T_read + T_pre + T_infer) + T_agg", size=24, weight="bold", fill="#1F3552")
    svg.text(x + 30, y + 126, "M = Σ I(r_m ≥ α),  α = 0.4", size=20, fill="#355C7D")
    svg.text(x + 780, y + 88, "P_slide,c = mean(TopK16(p_i,c))", size=22, weight="bold", fill="#1F3552")
    svg.text(x + 780, y + 126, "ŷ_c = I(P_slide,c ≥ τ_c)", size=20, fill="#355C7D")

    draw_box(
        svg,
        x + 24,
        y + 165,
        220,
        120,
        "WSI 读入\n金字塔 Level=1\nTile=224, Step=448",
        fill="#F4F8FB",
        stroke="#93A8BD",
    )
    draw_candidate_grid(svg, x + 305, y + 178)
    draw_box(
        svg,
        x + 610,
        y + 165,
        220,
        120,
        "有效 Tile 编码\nUNI / OM / ACL\n生成 tile-level prob",
        fill="#F4F8FB",
        stroke="#93A8BD",
    )
    draw_topk_bars(svg, x + 900, y + 193)
    draw_box(
        svg,
        x + 1170,
        y + 165,
        250,
        120,
        "TileAgg\nTopK mean\n输出 slide labels",
        fill="#F4F8FB",
        stroke="#93A8BD",
    )

    svg.line(x + 244, y + 225, x + 295, y + 225, stroke="#6C7A89", stroke_width=2.0, marker_end=True)
    svg.line(x + 450, y + 225, x + 605, y + 225, stroke="#6C7A89", stroke_width=2.0, marker_end=True)
    svg.line(x + 1080, y + 225, x + 1165, y + 225, stroke="#6C7A89", stroke_width=2.0, marker_end=True)

    config_text = "当前正式配置\nLevel=1 | tile=224 | step=448 | min_tissue=0.4\nmanifest cache=on | 聚合=TileAgg topk16"
    draw_box(svg, x + 24, y + 302, 690, 92, config_text, fill="#FAFCF7", stroke="#A4C89A", size=16)

    thr_text = (
        "类别阈值策略\n"
        f"Benign={thresholds['Benign']:.1f} | InSitu={thresholds['InSitu']:.1f} | Invasive={thresholds['Invasive']:.1f}\n"
        "若三类均未激活，则回退为 Normal"
    )
    draw_box(svg, x + 748, y + 302, 672, 92, thr_text, fill="#FFF9F2", stroke="#F0C48A", size=16)


def draw_axes(svg: SvgCanvas, x: float, y: float, w: float, h: float, *, xlabel: str, ylabel: str) -> None:
    svg.line(x, y, x, y + h, stroke="#2B3E50", stroke_width=1.8)
    svg.line(x, y + h, x + w, y + h, stroke="#2B3E50", stroke_width=1.8)
    svg.text(x + w / 2.0, y + h + 42, xlabel, size=18, anchor="middle")
    svg.text(x - 54, y + h / 2.0, ylabel, size=18, anchor="middle")


def draw_panel_b(svg: SvgCanvas, timing_frames: dict[str, list[dict]]) -> None:
    px = 40
    py = 470
    pw = 840
    ph = 560
    svg.rect(px, py, pw, ph, fill="#FCFDFE", stroke="#D6E0EA", stroke_width=1.2, rx=18)
    panel_header(svg, px + 16, py + 16, "B", "每张 WSI 耗时与有效 Tile 数近线性相关", 420)

    cx = px + 66
    cy = py + 82
    cw = 750
    ch = 410
    draw_axes(svg, cx, cy, cw, ch, xlabel="每张 Slide 的有效 Tile 数 M", ylabel="每张 Slide 总耗时 (s)")

    x_ticks = [100, 200, 300, 400]
    y_ticks = [5, 10, 15, 20, 25]
    for tick in x_ticks:
        tx = map_range(tick, 50, 480, cx, cx + cw)
        svg.line(tx, cy, tx, cy + ch, stroke="#D6E0EA", stroke_width=1.0, dash="5,5")
        svg.text(tx, cy + ch + 24, str(tick), size=14, anchor="middle", fill="#506070")
    for tick in y_ticks:
        ty = map_range(tick, 1.5, 26.5, cy + ch, cy)
        svg.line(cx, ty, cx + cw, ty, stroke="#D6E0EA", stroke_width=1.0, dash="5,5")
        svg.text(cx - 14, ty + 5, str(tick), size=14, anchor="end", fill="#506070")

    legend_x = px + 490
    legend_y = py + 110
    svg.rect(legend_x, legend_y, 300, 118, fill="white", stroke="#D0D7DE", stroke_width=1.0, rx=12)

    for idx, key in enumerate(["before", "origin", "mixed"]):
        cfg = RUN_CONFIGS[key]
        rows = timing_frames[key]
        xs = [float(row["n_tiles"]) for row in rows]
        ys = [float(row["elapsed_sec"]) for row in rows]
        slope, intercept, r2 = fit_line(xs, ys)
        pts = []
        for xv, yv in zip(xs, ys):
            sx = map_range(xv, 50, 480, cx, cx + cw)
            sy = map_range(yv, 1.5, 26.5, cy + ch, cy)
            pts.append((sx, sy))
            svg.circle(sx, sy, 5.5, fill=cfg["color"], stroke="white", stroke_width=1.0, opacity=0.9)
        line_xs = [60, 470]
        line_pts = []
        for xv in line_xs:
            yv = slope * xv + intercept
            sx = map_range(xv, 50, 480, cx, cx + cw)
            sy = map_range(yv, 1.5, 26.5, cy + ch, cy)
            line_pts.append((sx, sy))
        svg.polyline(line_pts, stroke=cfg["color"], stroke_width=3.0)

        ly = legend_y + 24 + idx * 30
        svg.line(legend_x + 14, ly, legend_x + 46, ly, stroke=cfg["color"], stroke_width=3.0)
        svg.circle(legend_x + 30, ly, 4.5, fill=cfg["color"], stroke="white", stroke_width=0.8)
        legend_text = f"{cfg['label']}  斜率={slope:.4f} s/tile,  R²={r2:.3f}"
        svg.text(legend_x + 58, ly + 5, legend_text, size=13)

    note_text = "A3 的关键不是只优化单次前向，\n而是降低 M，并减少每个 tile 的单位计算成本。"
    draw_box(svg, px + 455, py + 485, 340, 54, note_text, fill="white", stroke="#D0D7DE", size=14)


def draw_panel_c(svg: SvgCanvas, summaries: dict[str, dict]) -> None:
    px = 910
    py = 470
    pw = 650
    ph = 560
    svg.rect(px, py, pw, ph, fill="#FCFDFE", stroke="#D6E0EA", stroke_width=1.2, rx=18)
    panel_header(svg, px + 16, py + 16, "C", "正式 before/after 的性能前沿：更低总耗时，更高吞吐", 505)

    cx = px + 64
    cy = py + 82
    cw = 560
    ch = 410
    draw_axes(svg, cx, cy, cw, ch, xlabel="总耗时 (s)  ↓", ylabel="tiles/s  ↑")

    x_ticks = [0, 40, 80, 120, 160]
    y_ticks = [20, 40, 60, 80]
    for tick in x_ticks:
        tx = map_range(tick, 0, 170, cx, cx + cw)
        svg.line(tx, cy, tx, cy + ch, stroke="#D6E0EA", stroke_width=1.0, dash="5,5")
        svg.text(tx, cy + ch + 24, str(tick), size=14, anchor="middle", fill="#506070")
    for tick in y_ticks:
        ty = map_range(tick, 0, 98, cy + ch, cy)
        svg.line(cx, ty, cx + cw, ty, stroke="#D6E0EA", stroke_width=1.0, dash="5,5")
        svg.text(cx - 14, ty + 5, str(tick), size=14, anchor="end", fill="#506070")

    order = ["before", "origin", "mixed"]
    points: list[tuple[float, float]] = []
    for key in order:
        summary = summaries[key]
        x = map_range(float(summary["total_elapsed_sec"]), 0, 170, cx, cx + cw)
        y = map_range(float(summary["tiles_per_sec"]), 0, 98, cy + ch, cy)
        points.append((x, y))

    svg.polyline(points, stroke="#708090", stroke_width=2.2, opacity=0.9)
    for key, (sx, sy) in zip(order, points):
        summary = summaries[key]
        cfg = RUN_CONFIGS[key]
        radius = 12 + float(summary["wsi_per_sec"]) * 30
        svg.circle(sx, sy, radius, fill=cfg["color"], stroke="white", stroke_width=2.0, opacity=0.9)
        label = (
            f"{cfg['label']}\n"
            f"{summary['total_elapsed_sec']:.2f} s | {summary['avg_elapsed_sec_per_slide']:.2f} s/slide\n"
            f"{summary['wsi_per_sec']:.3f} WSI/s | {summary['tiles_per_sec']:.2f} tiles/s"
        )
        dx = 12
        dy = -38 if key == "before" else -46 if key == "origin" else -54
        draw_box(svg, sx + dx, sy + dy, 220, 74, label, fill="white", stroke="#D0D7DE", size=13)

    before = summaries["before"]
    origin = summaries["origin"]
    mixed = summaries["mixed"]
    speedup = float(before["total_elapsed_sec"]) / float(mixed["total_elapsed_sec"])
    runtime_drop = 1.0 - float(mixed["total_elapsed_sec"]) / float(before["total_elapsed_sec"])
    extra_gain = float(origin["total_elapsed_sec"]) / float(mixed["total_elapsed_sec"]) - 1.0
    note = (
        f"mixed OM 相对 before：{speedup:.2f}× 加速\n"
        f"总耗时下降：{runtime_drop * 100:.2f}%\n"
        f"相对 origin OM 再提速：{extra_gain * 100:.2f}%"
    )
    draw_box(svg, px + 352, py + 488, 248, 74, note, fill="#F7FBF4", stroke="#A4C89A", size=14)


def build_svg() -> str:
    summaries = {key: load_json(cfg["summary"]) for key, cfg in RUN_CONFIGS.items()}
    timing_frames = {key: load_csv(cfg["timing"]) for key, cfg in RUN_CONFIGS.items()}
    thresholds = load_json(THRESHOLDS_JSON)

    svg = SvgCanvas(SVG_W, SVG_H)
    svg.text(
        SVG_W / 2.0,
        42,
        "A3 任务级计算裁剪与聚合加速：从有效 Tile 数控制到正式 before/after 全流程收益",
        size=28,
        weight="bold",
        fill="#1F3552",
        anchor="middle",
    )
    svg.text(
        SVG_W / 2.0,
        72,
        "注：Panel B 使用官方 10 张无标签 WSI 的逐张实测耗时；Panel C 使用统一入口 run_submission_infer.py 的正式 before/origin OM/mixed OM 结果。",
        size=14,
        fill="#506070",
        anchor="middle",
    )
    draw_panel_a(svg, thresholds)
    draw_panel_b(svg, timing_frames)
    draw_panel_c(svg, summaries)
    return svg.to_svg()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "a3_tasklevel_acceleration_figure.svg"
    svg = build_svg()
    out_path.write_text(svg, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
