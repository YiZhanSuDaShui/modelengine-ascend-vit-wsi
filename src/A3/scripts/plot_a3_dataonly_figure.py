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
        "short": "before",
        "color": "#4C78A8",
        "summary": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/01_before_plain_pytorch/run/reports/run_summary.json",
        "timing": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/01_before_plain_pytorch/run/reports/per_slide_timing.csv",
    },
    "origin": {
        "label": "after origin OM",
        "short": "origin OM",
        "color": "#F58518",
        "summary": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/04_after_om_acl_sync/run/reports/run_summary.json",
        "timing": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/04_after_om_acl_sync/run/reports/per_slide_timing.csv",
    },
    "mixed": {
        "label": "after mixed OM",
        "short": "mixed OM",
        "color": "#54A24B",
        "summary": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/run/reports/run_summary.json",
        "timing": PROJECT_ROOT
        / "logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/run/reports/per_slide_timing.csv",
    },
}

SVG_W = 1680
SVG_H = 980
FONT_STACK = "Microsoft YaHei, SimHei, Arial, sans-serif"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(text: str) -> str:
    return escape(text)


class SvgCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items: list[str] = []

    def add(self, item: str) -> None:
        self.items.append(item)

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
        opacity: float | None = None,
        dash: str | None = None,
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
        if opacity is not None:
            attrs.append(f'opacity="{opacity:.3f}"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self.add(f"<rect {' '.join(attrs)} />")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        stroke_width: float = 1.0,
        dash: str | None = None,
        opacity: float | None = None,
        marker_end: bool = False,
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
        if opacity is not None:
            attrs.append(f'opacity="{opacity:.3f}"')
        if marker_end:
            attrs.append('marker-end="url(#arrow)"')
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
        line_height: float = 1.28,
    ) -> None:
        lines = fmt(text).split("\n")
        if len(lines) == 1:
            self.add(
                f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT_STACK}" font-size="{size}" '
                f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{lines[0]}</text>'
            )
            return
        tspans = []
        for idx, line in enumerate(lines):
            dy = 0 if idx == 0 else size * line_height
            tspans.append(f'<tspan x="{x:.2f}" dy="{dy:.2f}">{line}</tspan>')
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT_STACK}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">'
            + "".join(tspans)
            + "</text>"
        )

    def svg(self) -> str:
        defs = """
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L10,3 z" fill="#6C7A89" />
  </marker>
</defs>
"""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="white" />\n'
            f"{defs}\n"
            + "\n".join(self.items)
            + "\n</svg>\n"
        )


def map_range(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return dst_min
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


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


def panel_shell(svg: SvgCanvas, x: float, y: float, w: float, h: float, title: str) -> None:
    svg.rect(x, y, w, h, fill="#FCFDFE", stroke="#D7E1EA", stroke_width=1.2, rx=22)
    svg.rect(x + 20, y + 18, 260, 34, fill="#E2F0D9", stroke="none", rx=9)
    svg.text(x + 36, y + 41, title, size=18, weight="bold")


def draw_axes(svg: SvgCanvas, x: float, y: float, w: float, h: float, xlabel: str, ylabel: str) -> None:
    svg.line(x, y, x, y + h, stroke="#2B3E50", stroke_width=1.8)
    svg.line(x, y + h, x + w, y + h, stroke="#2B3E50", stroke_width=1.8)
    svg.text(x + w / 2.0, y + h + 46, xlabel, size=18, anchor="middle")
    svg.text(x - 58, y + h / 2.0, ylabel, size=18, anchor="middle")


def draw_scatter_panel(svg: SvgCanvas, x: float, y: float, w: float, h: float, timing_frames: dict[str, list[dict]]) -> None:
    panel_shell(svg, x, y, w, h, "A  Workload-Latency")
    plot_x = x + 78
    plot_y = y + 88
    plot_w = w - 120
    plot_h = h - 170
    draw_axes(svg, plot_x, plot_y, plot_w, plot_h, "每张 Slide 的有效 Tile 数 M", "每张 Slide 总耗时 (s)")

    x_ticks = [100, 200, 300, 400]
    y_ticks = [5, 10, 15, 20, 25]
    for tick in x_ticks:
        sx = map_range(tick, 50, 480, plot_x, plot_x + plot_w)
        svg.line(sx, plot_y, sx, plot_y + plot_h, stroke="#D6E0EA", stroke_width=1.0, dash="5,5")
        svg.text(sx, plot_y + plot_h + 24, str(tick), size=14, fill="#506070", anchor="middle")
    for tick in y_ticks:
        sy = map_range(tick, 1.5, 26.5, plot_y + plot_h, plot_y)
        svg.line(plot_x, sy, plot_x + plot_w, sy, stroke="#D6E0EA", stroke_width=1.0, dash="5,5")
        svg.text(plot_x - 12, sy + 5, str(tick), size=14, fill="#506070", anchor="end")

    legend_x = x + w - 360
    legend_y = y + 84
    svg.rect(legend_x, legend_y, 300, 118, fill="white", stroke="#D0D7DE", stroke_width=1.0, rx=12)

    for idx, key in enumerate(["before", "origin", "mixed"]):
        cfg = RUN_CONFIGS[key]
        rows = timing_frames[key]
        xs = [float(row["n_tiles"]) for row in rows]
        ys = [float(row["elapsed_sec"]) for row in rows]
        slope, intercept, r2 = fit_line(xs, ys)
        for xv, yv in zip(xs, ys):
            sx = map_range(xv, 50, 480, plot_x, plot_x + plot_w)
            sy = map_range(yv, 1.5, 26.5, plot_y + plot_h, plot_y)
            svg.circle(sx, sy, 6.0, fill=cfg["color"], stroke="white", stroke_width=1.0, opacity=0.88)
        line_pts = []
        for xv in [60, 470]:
            yv = slope * xv + intercept
            sx = map_range(xv, 50, 480, plot_x, plot_x + plot_w)
            sy = map_range(yv, 1.5, 26.5, plot_y + plot_h, plot_y)
            line_pts.append((sx, sy))
        svg.polyline(line_pts, stroke=cfg["color"], stroke_width=3.0)

        ly = legend_y + 24 + idx * 30
        svg.line(legend_x + 16, ly, legend_x + 46, ly, stroke=cfg["color"], stroke_width=3.0)
        svg.circle(legend_x + 31, ly, 4.5, fill=cfg["color"], stroke="white", stroke_width=0.8)
        svg.text(legend_x + 58, ly + 5, f"{cfg['short']} 斜率={slope:.4f} s/tile  |  R²={r2:.3f}", size=13)

    note = "同一批 WSI 在三种后端下保持相同的 tile 排序，\n斜率下降说明单位 tile 的端到端计算代价持续降低。"
    svg.rect(x + 430, y + h - 64, 350, 50, fill="white", stroke="#D0D7DE", stroke_width=1.0, rx=12)
    svg.text(x + 446, y + h - 40, note, size=14, fill="#455A64")


def draw_frontier_panel(svg: SvgCanvas, x: float, y: float, w: float, h: float, summaries: dict[str, dict]) -> None:
    panel_shell(svg, x, y, w, h, "B  Performance Frontier")
    plot_x = x + 74
    plot_y = y + 88
    plot_w = w - 108
    plot_h = h - 140
    draw_axes(svg, plot_x, plot_y, plot_w, plot_h, "总耗时 (s)  ↓", "tiles/s  ↑")

    x_ticks = [0, 40, 80, 120, 160]
    y_ticks = [20, 40, 60, 80]
    for tick in x_ticks:
        sx = map_range(tick, 0, 170, plot_x, plot_x + plot_w)
        svg.line(sx, plot_y, sx, plot_y + plot_h, stroke="#D6E0EA", stroke_width=1.0, dash="5,5")
        svg.text(sx, plot_y + plot_h + 24, str(tick), size=13, fill="#506070", anchor="middle")
    for tick in y_ticks:
        sy = map_range(tick, 0, 98, plot_y + plot_h, plot_y)
        svg.line(plot_x, sy, plot_x + plot_w, sy, stroke="#D6E0EA", stroke_width=1.0, dash="5,5")
        svg.text(plot_x - 12, sy + 4, str(tick), size=13, fill="#506070", anchor="end")

    order = ["before", "origin", "mixed"]
    pts: list[tuple[float, float]] = []
    for key in order:
        s = summaries[key]
        sx = map_range(float(s["total_elapsed_sec"]), 0, 170, plot_x, plot_x + plot_w)
        sy = map_range(float(s["tiles_per_sec"]), 0, 98, plot_y + plot_h, plot_y)
        pts.append((sx, sy))

    svg.polyline(pts, stroke="#708090", stroke_width=2.5, opacity=0.9)
    for key, (sx, sy) in zip(order, pts):
        cfg = RUN_CONFIGS[key]
        s = summaries[key]
        r = 14 + float(s["wsi_per_sec"]) * 35
        svg.circle(sx, sy, r, fill=cfg["color"], stroke="white", stroke_width=2.2, opacity=0.9)
        label = f"{cfg['short']}\n{float(s['total_elapsed_sec']):.2f}s | {float(s['tiles_per_sec']):.2f} tiles/s\n{float(s['wsi_per_sec']):.3f} WSI/s"
        bx = sx + 14 if key != "before" else sx - 188
        by = sy - 64 if key != "mixed" else sy - 86
        svg.rect(bx, by, 174, 58, fill="white", stroke="#D0D7DE", stroke_width=1.0, rx=12)
        svg.text(bx + 87, by + 18, label, size=13, anchor="middle")

    svg.text(x + w - 28, y + 70, "气泡面积 ∝ WSI/s", size=13, fill="#607080", anchor="end")


def draw_waterfall_panel(svg: SvgCanvas, x: float, y: float, w: float, h: float, summaries: dict[str, dict]) -> None:
    panel_shell(svg, x, y, w, h, "C  Runtime Waterfall")
    plot_x = x + 70
    plot_y = y + 88
    plot_w = w - 100
    plot_h = h - 150

    draw_axes(svg, plot_x, plot_y, plot_w, plot_h, "阶段", "总耗时 (s)")

    y_ticks = [0, 40, 80, 120, 160]
    for tick in y_ticks:
        sy = map_range(tick, 0, 170, plot_y + plot_h, plot_y)
        svg.line(plot_x, sy, plot_x + plot_w, sy, stroke="#D6E0EA", stroke_width=1.0, dash="5,5")
        svg.text(plot_x - 12, sy + 4, str(tick), size=13, fill="#506070", anchor="end")

    before = float(summaries["before"]["total_elapsed_sec"])
    origin = float(summaries["origin"]["total_elapsed_sec"])
    mixed = float(summaries["mixed"]["total_elapsed_sec"])
    delta_origin = before - origin
    delta_mixed = origin - mixed

    centers = [plot_x + 85, plot_x + 215, plot_x + 345, plot_x + 475, plot_x + 605]
    labels = ["before", "origin\n节省", "origin OM", "mixed\n再节省", "mixed OM"]
    for c, label in zip(centers, labels):
        svg.text(c, plot_y + plot_h + 34, label, size=14, fill="#405060", anchor="middle")

    def bar_top(v: float) -> float:
        return map_range(v, 0, 170, plot_y + plot_h, plot_y)

    # baseline
    bw = 70
    svg.rect(centers[0] - bw / 2, bar_top(before), bw, plot_y + plot_h - bar_top(before), fill="#4C78A8", stroke="none", rx=10, opacity=0.95)
    svg.text(centers[0], bar_top(before) - 10, f"{before:.2f}s", size=14, anchor="middle", weight="bold")

    # delta to origin
    svg.rect(centers[1] - bw / 2, bar_top(before), bw, bar_top(origin) - bar_top(before), fill="#F7D5B1", stroke="#F58518", stroke_width=1.2, rx=10, opacity=0.96)
    svg.text(centers[1], (bar_top(before) + bar_top(origin)) / 2 + 6, f"-{delta_origin:.2f}s", size=14, anchor="middle", weight="bold", fill="#B86210")

    svg.line(centers[0] + bw / 2 + 12, bar_top(before), centers[1] - bw / 2 - 12, bar_top(before), stroke="#95A5B2", stroke_width=1.6, dash="6,5")
    svg.line(centers[1] + bw / 2 + 12, bar_top(origin), centers[2] - bw / 2 - 12, bar_top(origin), stroke="#95A5B2", stroke_width=1.6, dash="6,5")

    # origin bar
    svg.rect(centers[2] - bw / 2, bar_top(origin), bw, plot_y + plot_h - bar_top(origin), fill="#F58518", stroke="none", rx=10, opacity=0.95)
    svg.text(centers[2], bar_top(origin) - 10, f"{origin:.2f}s", size=14, anchor="middle", weight="bold")

    # delta to mixed
    svg.rect(centers[3] - bw / 2, bar_top(origin), bw, bar_top(mixed) - bar_top(origin), fill="#CFE8C7", stroke="#54A24B", stroke_width=1.2, rx=10, opacity=0.98)
    svg.text(centers[3], (bar_top(origin) + bar_top(mixed)) / 2 + 6, f"-{delta_mixed:.2f}s", size=14, anchor="middle", weight="bold", fill="#3F7F38")

    svg.line(centers[2] + bw / 2 + 12, bar_top(origin), centers[3] - bw / 2 - 12, bar_top(origin), stroke="#95A5B2", stroke_width=1.6, dash="6,5")
    svg.line(centers[3] + bw / 2 + 12, bar_top(mixed), centers[4] - bw / 2 - 12, bar_top(mixed), stroke="#95A5B2", stroke_width=1.6, dash="6,5")

    # mixed bar
    svg.rect(centers[4] - bw / 2, bar_top(mixed), bw, plot_y + plot_h - bar_top(mixed), fill="#54A24B", stroke="none", rx=10, opacity=0.97)
    svg.text(centers[4], bar_top(mixed) - 10, f"{mixed:.2f}s", size=14, anchor="middle", weight="bold")

    speedup = before / mixed
    runtime_drop = (1.0 - mixed / before) * 100.0
    extra_gain = (origin / mixed - 1.0) * 100.0
    note = f"mixed OM 相对 before：{speedup:.2f}×\n总耗时下降：{runtime_drop:.2f}%\n相对 origin 再提速：{extra_gain:.2f}%"
    svg.rect(x + w - 242, y + h - 94, 210, 70, fill="white", stroke="#D0D7DE", stroke_width=1.0, rx=12)
    svg.text(x + w - 137, y + h - 68, note, size=13, anchor="middle")


def build_svg() -> str:
    summaries = {key: load_json(cfg["summary"]) for key, cfg in RUN_CONFIGS.items()}
    timing_frames = {key: load_csv(cfg["timing"]) for key, cfg in RUN_CONFIGS.items()}

    svg = SvgCanvas(SVG_W, SVG_H)
    svg.text(
        SVG_W / 2.0,
        46,
        "A3：WSI 任务级计算裁剪与聚合加速",
        size=30,
        weight="bold",
        fill="#1F3552",
        anchor="middle",
    )
    svg.text(
        SVG_W / 2.0,
        76,
        "同口径 before / origin OM / mixed OM 数据图，仅保留任务级 workload 与正式性能结果",
        size=15,
        fill="#607080",
        anchor="middle",
    )

    draw_scatter_panel(svg, 40, 110, 950, 820, timing_frames)
    draw_frontier_panel(svg, 1020, 110, 620, 380, summaries)
    draw_waterfall_panel(svg, 1020, 530, 620, 400, summaries)
    return svg.svg()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "a3_tasklevel_dataonly_figure.svg"
    out_path.write_text(build_svg(), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
