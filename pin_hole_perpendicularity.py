#!/usr/bin/env python3
"""计算针齿孔轴线相对端面法向的垂直度。

点云先拟合盘面，把法向转到 +Z，原点放在盘面中心。针齿孔是内壁上向外鼓出的
圆弧槽。整段直壁一起拟合出针齿半径。针齿圆心落在内壁上，每一层用该齿左右
内壁的半径作为圆心半径。

垂直度用这些圆心从直壁下沿到上沿的横移来算：

- 径向横移按方位呈一次谐波。幅度除以高度，就是整圈针齿孔轴线相对端面法向的倾斜。
- 切向看 31 孔整体相位随高度的变化，是整圈扭转。

示例：

    python pin_hole_perpendicularity.py --scan 外壳完整点云.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from compare_machining_error import fit_disc_plane, read_ply
from pin_profile_error import (
    analyze_xy,
    extract_edge,
    fourier_phase,
    wrap,
)


def _cjk(size: float) -> font_manager.FontProperties:
    """返回指定字号的中文字体。"""
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(path).is_file():
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path, size=size)
    return font_manager.FontProperties(size=size)


def face_frame(points: np.ndarray) -> np.ndarray:
    """盘面法向转到 +Z。返回的 z 就是相对拟合盘面的高度，毫米。"""
    center, _e1, _e2, normal, _height = fit_disc_plane(points)
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z_axis)
    sine = float(np.linalg.norm(v))
    cosine = float(np.dot(normal, z_axis))
    if sine < 1e-12:
        rotation = np.eye(3) if cosine > 0.0 else np.diag([1.0, -1.0, -1.0])
    else:
        cross = np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
            dtype=float,
        )
        rotation = np.eye(3) + cross + cross @ cross * ((1.0 - cosine) / (sine * sine))
    return (points - center) @ rotation.T


def _edge(xy: np.ndarray, n_bins: int, r_lo: float, r_hi: float):
    """在给定半径带里提取内轮廓，并把没有点的角度用圆周插值补上。"""
    theta, radius, valid = extract_edge(xy, r_lo, r_hi, n_bins)
    if int(valid.sum()) < n_bins // 3:
        raise RuntimeError("这一层针齿轮廓点太少")
    curve = radius.copy()
    if (~valid).any():
        index = np.arange(len(curve))
        curve[~valid] = np.interp(index[~valid], index[valid], curve[valid], period=len(curve))
    return theta, curve, valid


def pin_angles(theta: np.ndarray, radius: np.ndarray, pins: int) -> np.ndarray:
    """半径波峰的方位。波峰是针齿槽向外鼓出的位置。"""
    phase = fourier_phase(theta, radius, pins)
    angles = phase + np.arange(pins) * (2.0 * np.pi / pins)
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def center_radii(
    theta: np.ndarray,
    radius: np.ndarray,
    angles: np.ndarray,
    half_width: float,
) -> np.ndarray:
    """每个针齿圆心所在的内壁半径。

    针齿圆弧向外鼓出，圆心落在左右内壁上。在半个齿距以内取内轮廓的最小半径，
    就是这段内壁。
    """
    centers = np.empty(len(angles))
    for i, alpha in enumerate(angles):
        mine = np.abs(wrap(theta - alpha)) <= half_width
        centers[i] = float(np.min(radius[mine]))
    return centers


def straight_wall(uv: np.ndarray, pins: int, n_bins: int, r_lo: float, r_hi: float) -> tuple[float, float]:
    """31 齿谐波一直够强的那段高度，就是针齿直壁。"""
    z = uv[:, 2]
    lo, hi = np.percentile(z, [2, 98])
    edges = np.arange(lo, hi, 1.0)
    strong = []
    for z0 in edges:
        pts = uv[(z >= z0) & (z < z0 + 1.0), :2]
        if len(pts) < n_bins:
            strong.append(False)
            continue
        try:
            theta, curve, valid = _edge(pts, n_bins, r_lo, r_hi)
        except RuntimeError:
            strong.append(False)
            continue
        signal = curve - curve.mean()
        amplitude = 2.0 * abs(np.fft.rfft(signal)[pins]) / len(signal)
        strong.append(bool(amplitude > 0.8 and valid.mean() > 0.45))
    strong = np.asarray(strong)
    if not strong.any():
        raise RuntimeError("点云里没有找到带 31 个针齿槽的直壁")
    best = (0, 1)
    start = None
    for i, flag in enumerate(np.r_[strong, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return float(edges[best[0]]), float(edges[best[1] - 1] + 1.0)


def measure(uv: np.ndarray, pins: int, n_bins: int, step: float) -> dict:
    """沿针齿直壁逐层取 31 个圆心半径和整圈相位，据此计算整圈轴线倾斜和整圈扭转。"""
    radius = np.hypot(uv[:, 0], uv[:, 1])
    band = (radius > 38.0) & (radius < 49.0)
    r_lo, r_hi = 40.0, 47.5
    z0, z1 = straight_wall(uv[band], pins, n_bins, r_lo, r_hi)
    margin = 1.2
    if z1 - z0 > 2.0 * margin + 4.0:
        z0, z1 = z0 + margin, z1 - margin
    reference = uv[(uv[:, 2] >= (z0 + z1) / 2 - 2.0) & (uv[:, 2] < (z0 + z1) / 2 + 2.0), :2]
    fitted = analyze_xy(reference, None, n_bins, pins)
    pitch_radius = float(fitted["fit"]["pitch_radius"])
    pin_radius = float(fitted["fit"]["pin_radius"])

    ref_theta, ref_curve, _ref_valid = _edge(reference, n_bins, r_lo, r_hi)
    angles = np.sort(pin_angles(ref_theta, ref_curve, pins))
    half_width = np.pi / pins * 0.85

    z_edges = np.arange(z0, z1 - step * 0.5, step)
    heights = []
    centers = []
    phases = []
    for z_slice in z_edges:
        pts = uv[(uv[:, 2] >= z_slice) & (uv[:, 2] < z_slice + step), :2]
        if len(pts) < n_bins:
            continue
        try:
            theta, curve, _valid = _edge(pts, n_bins, r_lo, r_hi)
        except RuntimeError:
            continue
        heights.append(z_slice + step / 2)
        centers.append(center_radii(theta, curve, angles, half_width))
        phases.append(fourier_phase(theta, curve, pins))
    if len(heights) < 4:
        raise RuntimeError("直壁上能够拟合针齿圆心的层数太少")

    heights = np.asarray(heights)
    centers = np.vstack(centers)
    phases = np.unwrap(np.asarray(phases))
    dz = float(heights[-1] - heights[0])
    radial = np.full(pins, np.nan)
    for i in range(pins):
        ok = np.isfinite(centers[:, i])
        if int(ok.sum()) < 4:
            continue
        slope = float(np.polyfit(heights[ok], centers[ok, i], 1)[0])
        radial[i] = slope * dz
    ok = np.isfinite(radial)
    design = np.column_stack([np.cos(angles[ok]), np.sin(angles[ok]), np.ones(int(ok.sum()))])
    coef, *_ = np.linalg.lstsq(design, radial[ok], rcond=None)
    amplitude = float(np.hypot(coef[0], coef[1]))
    tilt = float(np.degrees(np.arctan(amplitude / dz)))
    direction = float(np.degrees(np.arctan2(coef[1], coef[0])))
    phase_fit = np.polyval(np.polyfit(heights, phases, 1), heights)
    twist = float(np.degrees(phase_fit[-1] - phase_fit[0]))
    tangential = float(pitch_radius * np.radians(twist))
    return {
        "uv": uv,
        "z0": z0,
        "z1": z1,
        "pitch_radius": pitch_radius,
        "pin_radius": pin_radius,
        "angles": angles,
        "heights": heights,
        "centers": centers,
        "phases": phases,
        "radial": radial,
        "amplitude": amplitude,
        "tilt": tilt,
        "direction": direction,
        "mean_radial": float(coef[2]),
        "dz": dz,
        "twist": twist,
        "tangential": tangential,
        "residual": float(np.std(radial[ok] - design @ coef)),
    }


def write_report(path: Path, result: dict) -> None:
    """把垂直度结果写成文字，并列出每个针齿孔从下到上的径向位移。"""
    angles = result["angles"]
    radial = result["radial"]
    order = np.argsort(angles)
    lines = [
        "针齿孔轴线相对端面法向的垂直度。",
        "坐标：盘面法向为 +Z，原点在拟合盘面中心。单位毫米、度。",
        f"直壁 z {result['z0']:.1f} 到 {result['z1']:.1f} mm，圆心比较用的高度差 {result['dz']:.1f} mm。",
        f"针齿中心圆半径 {result['pitch_radius']:.3f} mm，针齿半径 {result['pin_radius']:.3f} mm。",
        "",
        f"整圈倾斜 {result['tilt']:.3f}°，径向幅度 {result['amplitude']:.4f} mm，偏向方位 {result['direction']:.1f}°。",
        f"31 孔一起变深或变浅 {result['mean_radial']:+.4f} mm。去掉整圈倾斜后，单孔径向偏移标准差 {result['residual']:.4f} mm。",
        f"整圈扭转 {result['twist']:+.4f}°，折到针齿圆心 {result['tangential']:+.4f} mm。",
        "",
        "编号, 方位°, 径向偏移mm",
    ]
    for n, i in enumerate(order, start=1):
        lines.append(f"{n:2d}, {np.degrees(angles[i]):7.2f}, {radial[i]:+.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pin_circle(center_r: float, alpha: float, pin_radius: float, phi: np.ndarray) -> np.ndarray:
    """生成一颗针齿圆上的采样点。圆心在该齿径向上，半径为针齿半径。"""
    radial_hat = np.array([np.cos(alpha), np.sin(alpha)])
    tangential_hat = np.array([-np.sin(alpha), np.cos(alpha)])
    center = center_r * radial_hat
    return center + pin_radius * (
        np.cos(phi)[:, None] * radial_hat + np.sin(phi)[:, None] * tangential_hat
    )


def plot_report(path: Path, result: dict) -> None:
    """画最下层和最上层的针齿圆，以及整圈倾斜、整圈扭转两张曲线。"""
    uv = result["uv"]
    angles = result["angles"]
    heights = result["heights"]
    pin_radius = float(result["pin_radius"])
    wall = (uv[:, 2] >= result["z0"]) & (uv[:, 2] < result["z1"])
    rel = uv[wall, :2]
    order = np.argsort(angles)
    radial = result["radial"][order]
    theta = angles[order]
    design = np.column_stack([np.cos(theta), np.sin(theta), np.ones(len(theta))])
    coef, *_ = np.linalg.lstsq(design, radial, rcond=None)
    half = float(np.median(np.diff(heights))) / 2.0
    layers = (
        (0, "#1a73e8", f"最下层 z {heights[0]:.1f} mm"),
        (-1, "#e37400", f"最上层 z {heights[-1]:.1f} mm"),
    )
    phi = np.linspace(0.0, 2.0 * np.pi, 90)

    fig = plt.figure(figsize=(12.2, 8.2))
    ax0 = fig.add_axes([0.05, 0.50, 0.46, 0.42])
    step = max(1, len(rel) // 50000)
    ax0.scatter(rel[::step, 0], rel[::step, 1], s=0.2, c="#d0d0d0", linewidths=0, rasterized=True)
    for index, color, _label in layers:
        for alpha, center_r in zip(angles, result["centers"][index]):
            if not np.isfinite(center_r):
                center_r = result["pitch_radius"]
            circle = _pin_circle(float(center_r), float(alpha), pin_radius, phi)
            ax0.plot(circle[:, 0], circle[:, 1], color=color, lw=0.8)
    ax0.set_aspect("equal")
    ax0.set_xlabel("x (mm)", fontproperties=_cjk(11))
    ax0.set_ylabel("y (mm)", fontproperties=_cjk(11))
    ax0.set_title("最下层与最上层的针齿圆", fontproperties=_cjk(13))
    ax0.tick_params(labelsize=9)
    ax0.legend(
        handles=[
            plt.Line2D([0], [0], color=color, lw=1.6, label=label)
            for _index, color, label in layers
        ],
        prop=_cjk(9),
        loc="upper right",
        frameon=False,
    )

    focus = int(np.argmin(np.abs(angles)))
    alpha = float(angles[focus])
    radial_hat = np.array([np.cos(alpha), np.sin(alpha)])
    tangential_hat = np.array([-np.sin(alpha), np.cos(alpha)])
    tip = (float(np.nanmean(result["centers"][:, focus])) + pin_radius) * radial_hat
    axz = fig.add_axes([0.05, 0.07, 0.46, 0.34])
    for index, color, _label in layers:
        z_slice = float(heights[index])
        band = np.abs(uv[:, 2] - z_slice) <= half
        xy = uv[band, :2]
        local = xy - tip
        tang = local @ tangential_hat
        rad = local @ radial_hat
        keep = (np.abs(tang) < 1.6) & (rad > -1.8) & (rad < 0.9)
        axz.scatter(tang[keep], rad[keep], s=6, c=color, linewidths=0, alpha=0.45, rasterized=True)
        center_r = result["centers"][index, focus]
        if not np.isfinite(center_r):
            center_r = result["pitch_radius"]
        circle = _pin_circle(float(center_r), alpha, pin_radius, phi)
        local_c = circle - tip
        axz.plot(local_c @ tangential_hat, local_c @ radial_hat, color=color, lw=1.4)
    axz.set_aspect("equal")
    axz.set_xlabel("切向 (mm)", fontproperties=_cjk(11))
    axz.set_ylabel("径向向外 (mm)", fontproperties=_cjk(11))
    axz.set_title(
        "同一颗针齿圆放大，两层圆心相差 "
        f"{(result['centers'][-1, focus] - result['centers'][0, focus]) * 1000:.0f} µm",
        fontproperties=_cjk(12),
    )
    axz.tick_params(labelsize=9)

    ax1 = fig.add_axes([0.60, 0.56, 0.35, 0.32])
    ax1.bar(np.arange(1, len(radial) + 1), radial, color="#188038")
    ax1.plot(np.arange(1, len(radial) + 1), design @ coef, color="#202124", lw=1.2)
    ax1.axhline(0.0, color="#202124", lw=0.6)
    ax1.set_xlabel("针齿孔编号（从 +x 起逆时针）", fontproperties=_cjk(11))
    ax1.set_ylabel("圆心径向偏移 (mm)", fontproperties=_cjk(11))
    ax1.set_title(
        f"整圈轴线倾斜 {result['tilt']:.2f}°，径向幅度 {result['amplitude']:.3f} mm",
        fontproperties=_cjk(12),
    )
    ax1.tick_params(labelsize=8)
    ax1.set_xticks([1, 6, 11, 16, 21, 26, 31][: len(radial)])

    ax2 = fig.add_axes([0.60, 0.10, 0.35, 0.30])
    degrees = np.degrees(result["phases"])
    fit = np.polyval(np.polyfit(result["heights"], degrees, 1), result["heights"])
    ax2.plot(result["heights"], degrees, "o", color="#188038", ms=4)
    ax2.plot(result["heights"], fit, color="#202124", lw=1.2)
    ax2.set_xlabel("高度 z (mm)", fontproperties=_cjk(11))
    ax2.set_ylabel("31 孔整体相位 (°)", fontproperties=_cjk(11))
    ax2.set_title(
        f"整圈扭转 {result['twist']:+.3f}°，折合孔位 {result['tangential']:+.3f} mm",
        fontproperties=_cjk(12),
    )
    ax2.tick_params(labelsize=9)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数：点云、针齿数、角度采样、分层厚度和输出目录。"""
    parser = argparse.ArgumentParser(description="计算针齿孔轴线相对端面法向的垂直度")
    parser.add_argument("--scan", type=Path, default=Path("外壳完整点云.ply"), help="外壳点云 PLY")
    parser.add_argument("--pins", type=int, default=31, help="针齿孔数量，默认 31")
    parser.add_argument("--bins", type=int, default=1440, help="每一层轮廓的角度采样数")
    parser.add_argument("--step", type=float, default=0.8, help="分层厚度，毫米")
    parser.add_argument("--out", type=Path, default=Path("正面视角"), help="结果目录")
    return parser


def main(argv: list[str] | None = None) -> None:
    """读外壳点云，计算针齿孔垂直度，并写出图和文字结果。"""
    args = build_parser().parse_args(argv)
    if not args.scan.is_file():
        raise SystemExit(f"找不到测量点云: {args.scan}")
    points, _faces = read_ply(args.scan)
    result = measure(face_frame(points), args.pins, args.bins, args.step)
    args.out.mkdir(parents=True, exist_ok=True)
    write_report(args.out / "针齿孔垂直度.txt", result)
    plot_report(args.out / "外齿廓_垂直度.png", result)
    print(
        f"直壁 z {result['z0']:.1f} 到 {result['z1']:.1f} mm，高度差 {result['dz']:.1f} mm\n"
        f"针齿中心圆 {result['pitch_radius']:.3f} mm，针齿半径 {result['pin_radius']:.3f} mm\n"
        f"整圈倾斜 {result['tilt']:.3f}°，径向幅度 {result['amplitude']:.4f} mm，偏向 {result['direction']:.1f}°\n"
        f"整圈扭转 {result['twist']:+.4f}°，折合 {result['tangential']:+.4f} mm\n"
        f"结果写在 {args.out}"
    )


if __name__ == "__main__":
    main()
