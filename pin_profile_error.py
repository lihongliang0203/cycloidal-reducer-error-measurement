#!/usr/bin/env python3
"""根据点云计算摆线减速机针齿齿廓误差和节圆半径误差。

针齿齿廓是外圈内壁上一圈圆弧，圆心落在节圆上。程序从测量点云里提出这圈
齿廓，拟合每颗针齿的圆心和半径，再和设计值比较。

误差定义（长度单位毫米）：

- 节圆半径误差 = 实测针齿中心圆半径 − 设计节圆半径。正值表示针齿中心整体偏外。
- 针齿径向偏差 = 该针齿圆心到回转中心的距离 − 设计节圆半径。
- 齿廓偏差 = 实测点到该针齿拟合圆心的距离 − 拟合针齿半径。
  正值表示测点离圆心比针齿半径更远，齿槽偏大。
- 形状偏差用同一批边缘点，但半径换成这颗针齿自己的拟合半径，用来看圆度。

设计值来自三维模型，或来自 --pins、--pitch-diameter、--pin-diameter。
只给点云时，以测量自身拟合出的节圆为基准，结果表示各针齿相对这条节圆的不一致。

示例：

    python pin_profile_error.py --scan 2.ply --model 外圈.stl
    python pin_profile_error.py --scan 2.ply --pins 31 --pitch-diameter 85 --pin-diameter 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_machining_error import load_model, read_ply  # noqa: E402


def _setup_font() -> None:
    """注册中文字体，并让坐标轴负号按普通减号显示。"""
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    ):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


_setup_font()


def _cjk_font(size: float) -> font_manager.FontProperties:
    """返回指定字号的中文字体。"""
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(path).is_file():
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path, size=size)
    return font_manager.FontProperties(family="Noto Sans CJK JP", size=size)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def rotate(pts: np.ndarray, beta: float) -> np.ndarray:
    """把平面点绕原点旋转 beta 弧度。"""
    c, s = np.cos(beta), np.sin(beta)
    x, y = pts[:, 0], pts[:, 1]
    return np.column_stack([c * x - s * y, s * x + c * y])


def wrap(angle: np.ndarray) -> np.ndarray:
    """把角度折到 (−π, π]。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pin_index(angle: np.ndarray, pins: int) -> np.ndarray:
    """按方位把角度归到最近的一颗针齿，编号从 0 起。"""
    pitch = 2 * np.pi / pins
    rel = np.mod(angle, 2 * np.pi)
    return np.rint(rel / pitch).astype(int) % pins


def half_span(pitch_radius: float, pin_radius: float) -> float:
    """针齿圆弧在极角上的半宽。超过这个角度，轮廓回到节圆。"""
    if pitch_radius <= pin_radius:
        raise ValueError("节圆半径必须大于针齿半径")
    return float(np.arcsin(pin_radius / pitch_radius))


def fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """代数初值后再做几何拟合。返回圆心和半径。"""
    x, y = pts[:, 0], pts[:, 1]
    mat = np.column_stack([2 * x, 2 * y, np.ones(len(pts))])
    cx, cy, c0 = np.linalg.lstsq(mat, x * x + y * y, rcond=None)[0]
    center = np.array([cx, cy], dtype=float)
    radius = float(np.sqrt(max(c0 + cx * cx + cy * cy, 0.0)))
    for _ in range(12):
        dvec = pts - center
        dist = np.maximum(np.hypot(dvec[:, 0], dvec[:, 1]), 1e-9)
        res = dist - radius
        jac = np.column_stack([-dvec[:, 0] / dist, -dvec[:, 1] / dist, -np.ones(len(pts))])
        step, *_ = np.linalg.lstsq(jac, -res, rcond=None)
        center = center + step[:2]
        radius = float(max(0.05, radius + step[2]))
        if np.linalg.norm(step) < 1e-7:
            break
    return center, radius


def circular_median(values: np.ndarray, width: int = 3) -> np.ndarray:
    """沿圆周做滑动中值滤波，首尾相接。"""
    if width % 2 == 0:
        width += 1
    half = width // 2
    padded = np.r_[values[-half:], values, values[:half]]
    view = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(view, axis=1)


def huber_weight(residual: np.ndarray, delta: float = 0.2) -> np.ndarray:
    """大残差降权，返回加权最小二乘用的平方根权重。"""
    scale = np.abs(residual)
    weight = np.ones_like(scale)
    big = scale > delta
    weight[big] = delta / np.maximum(scale[big], 1e-12)
    return np.sqrt(weight)


def robust_rms(residual: np.ndarray) -> float:
    """去掉离群残差后计算均方根。"""
    if len(residual) == 0:
        return float("inf")
    med = np.median(residual)
    keep = np.abs(residual - med) < max(0.5, 6 * (np.median(np.abs(residual - med)) + 1e-6))
    use = residual[keep] if keep.sum() >= max(8, len(residual) // 5) else residual
    return float(np.sqrt(np.mean(use ** 2)))


def stats(values: np.ndarray) -> dict:
    """计算有限值的均值、标准差、均方根、极值和峰谷。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {k: float("nan") for k in ("mean", "std", "rms", "min", "max", "pv")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "pv": float(np.ptp(values)),
    }


# ---------------------------------------------------------------------------
# 从平面点云提取针齿内轮廓
# ---------------------------------------------------------------------------

def _histogram_bands(radius: np.ndarray) -> list[tuple[float, float]]:
    """按半径把点分成几段密集环带。针齿、内孔、外圆会落在不同的带里。"""
    hi = float(np.percentile(radius, 99.5))
    if hi <= 0:
        return [(0.0, 1.0)]
    edges = np.linspace(0.0, hi * 1.02, 90)
    hist, edges = np.histogram(radius, bins=edges)
    smooth = np.convolve(hist.astype(float), np.array([1, 2, 3, 2, 1], dtype=float) / 9, mode="same")
    high = smooth >= max(8.0, 0.12 * float(smooth.max()))
    bands: list[tuple[float, float]] = []
    i = 0
    while i < len(high):
        if not high[i]:
            i += 1
            continue
        j = i
        while j < len(high) and high[j]:
            j += 1
        bands.append((float(edges[i]), float(edges[j])))
        i = j
    if not bands:
        bands.append((0.0, hi))
    return bands


def _envelope_at_center(rel: np.ndarray, radius: np.ndarray) -> float:
    """陡齿侧上，按角度取内包络再插值到扇区中心，避免被扇区内侧的点拉小。"""
    floor = float(np.min(radius))
    keep = radius <= floor + 0.45
    if int(keep.sum()) < 2:
        return floor
    x = rel[keep]
    y = radius[keep]
    key = np.round(x, 5)
    xs = []
    ys = []
    for item in np.unique(key):
        sel = key == item
        xs.append(float(np.mean(x[sel])))
        ys.append(float(np.min(y[sel])))
    xs_a = np.asarray(xs)
    ys_a = np.asarray(ys)
    order = np.argsort(xs_a)
    xs_a, ys_a = xs_a[order], ys_a[order]
    left = xs_a < 0.0
    right = xs_a >= 0.0
    if left.any() and right.any():
        i_left = int(np.argmax(xs_a[left]))
        i_right = int(np.argmin(xs_a[right]))
        x_left, y_left = float(xs_a[left][i_left]), float(ys_a[left][i_left])
        x_right, y_right = float(xs_a[right][i_right]), float(ys_a[right][i_right])
        if x_right > x_left:
            t = (0.0 - x_left) / (x_right - x_left)
            return float(y_left + t * (y_right - y_left))
    return float(np.min(ys_a))


def _inner_radius(angle: np.ndarray, radius: np.ndarray, theta: float) -> float:
    """扇区内侧边界在扇区中心角上的半径。

    齿底和齿间圆弧上，边界几乎不随角度变，用最内侧且跨过扇区中心的点。
    齿侧很陡时，扇区里更靠内的那一侧会把半径拉小；结构光点云在齿侧仍然准，
    所以改读内包络在扇区中心的值。
    """
    rel = wrap(angle - theta)
    low = np.percentile(radius, 8) if len(radius) >= 25 else float(np.min(radius))
    band = radius <= low + 0.15
    if int(band.sum()) < 2:
        return float(np.min(radius))
    x = rel[band]
    y = radius[band]
    if float(np.max(x) - np.min(x)) < 1e-8:
        return float(np.min(y))
    weight = np.exp(-(y - np.min(y)) / 0.03)
    design = np.column_stack([np.ones(len(x)), x]) * np.sqrt(weight)[:, None]
    coef, *_ = np.linalg.lstsq(design, y * np.sqrt(weight), rcond=None)
    inner = weight > 0.2
    if np.any(inner) and np.min(x[inner]) < 0.0 < np.max(x[inner]):
        return float(coef[0])
    return _envelope_at_center(rel, radius)


def extract_edge(xy: np.ndarray, r_lo: float, r_hi: float, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回扇区中心角、内轮廓半径、以及半径有效的掩码。"""
    radius = np.hypot(xy[:, 0], xy[:, 1])
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    use = (radius >= r_lo) & (radius <= r_hi)
    if int(use.sum()) < n_bins:
        raise RuntimeError("针齿半径带里的点太少，无法提取齿廓")
    theta = (np.arange(n_bins) + 0.5) / n_bins * 2 * np.pi - np.pi
    bins = np.floor((angle[use] + np.pi) / (2 * np.pi) * n_bins).astype(np.int64)
    bins = np.clip(bins, 0, n_bins - 1)
    order = np.argsort(bins, kind="mergesort")
    bins_s = bins[order]
    rad_s = radius[use][order]
    ang_s = angle[use][order]
    rmin = np.full(n_bins, np.nan)
    cuts = np.flatnonzero(np.diff(bins_s)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(bins_s)]
    for b, s, e in zip(bins_s[starts], starts, ends):
        if e - s < 2:
            continue
        rmin[b] = _inner_radius(ang_s[s:e], rad_s[s:e], theta[b])
    valid = np.isfinite(rmin)
    valid = _drop_outer_spikes(rmin, valid)
    return theta, rmin, valid


def _drop_outer_spikes(rmin: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """丢掉个别角度上内轮廓缺失、半径跳到更外一圈轮廓的点。

    针齿槽本身有两三个毫米高，但宽度接近一个齿距，不会被这里当成尖峰。
    """
    if int(valid.sum()) < 30:
        return valid
    curve = rmin.copy()
    idx = np.arange(len(curve))
    if (~valid).any():
        curve[~valid] = np.interp(idx[~valid], idx[valid], curve[valid], period=len(curve))
    # 只去掉比左右邻域突然高出一截的窄尖峰。齿槽有几个毫米宽，不会被抹掉。
    local = circular_median(curve, 7)
    bad = valid & (rmin > local + 1.5)
    cleaned = valid.copy()
    cleaned[bad] = False
    if int(cleaned.sum()) < int(valid.sum()) * 0.5:
        return valid
    return cleaned


def _pin_score(rmin: np.ndarray, valid: np.ndarray) -> tuple[float, int, float]:
    """用半径曲线的主谐波判断这是不是针齿齿廓。返回 (幅值, 针齿数, 峰谷)。"""
    if valid.mean() < 0.55:
        return 0.0, 0, 0.0
    curve = rmin.copy()
    idx = np.arange(len(curve))
    if (~valid).any():
        curve[~valid] = np.interp(idx[~valid], idx[valid], curve[valid], period=len(curve))
    span = float(np.percentile(curve, 98) - np.percentile(curve, 10))
    if not 0.35 <= span <= 15.0:
        return 0.0, 0, span
    spec = np.fft.rfft(curve - curve.mean())
    mag = np.abs(spec)
    lo, hi = 8, min(80, len(mag) - 1)
    if hi <= lo:
        return 0.0, 0, span
    k = lo + int(np.argmax(mag[lo:hi]))
    amplitude = 2.0 * float(mag[k]) / len(curve)
    if amplitude < 0.12:
        return 0.0, 0, span
    return amplitude, k, span


def fourier_phase(theta: np.ndarray, radius: np.ndarray, pins: int) -> float:
    """半径曲线相对 cos(pins·θ) 的相位。波峰在 θ = phase + 2πk/pins。"""
    signal = radius - np.mean(radius)
    coeff_cos = float(np.sum(signal * np.cos(pins * theta)))
    coeff_sin = float(np.sum(signal * np.sin(pins * theta)))
    return float(np.arctan2(coeff_sin, coeff_cos) / pins)


def _fill(theta: np.ndarray, rmin: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """把内轮廓上空缺的角度用圆周插值补上。"""
    curve = rmin.copy()
    if (~valid).any() and valid.any():
        idx = np.arange(len(curve))
        curve[~valid] = np.interp(idx[~valid], idx[valid], curve[valid], period=len(curve))
    return curve


def select_profile(xy: np.ndarray, n_bins: int, pins: int | None = None) -> dict:
    """在若干半径带里挑出针齿内轮廓，并把节圆中心移到原点、一号齿转到 0°。"""
    center = np.median(xy, axis=0)
    chosen = None
    for _ in range(4):
        rel = xy - center
        radius = np.hypot(rel[:, 0], rel[:, 1])
        bands = _histogram_bands(radius)
        bands.append((float(np.percentile(radius, 1)), float(np.percentile(radius, 99))))
        best = None
        for lo, hi in bands:
            try:
                theta, rmin, valid = extract_edge(rel, lo - 0.4, hi + 0.8, n_bins)
            except RuntimeError:
                continue
            score, found, span = _pin_score(rmin, valid)
            if pins is not None and found != pins:
                # 设计针齿数已知时，仍要求这条曲线在该齿数上有足够谐波
                curve = _fill(theta, rmin, valid)
                spec = np.abs(np.fft.rfft(curve - curve.mean()))
                amp = 2.0 * float(spec[pins]) / len(curve) if pins < len(spec) else 0.0
                if amp < 0.12 or not 0.35 <= span <= 15:
                    continue
                score, found = amp, pins
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "pins": found,
                    "span": span,
                    "theta": theta,
                    "rmin": rmin,
                    "valid": valid,
                    "lo": lo - 0.4,
                    "hi": hi + 0.8,
                }
        if best is None or best["pins"] < 8:
            raise RuntimeError("点云里没有找到整圈针齿齿廓。请确认测的是针齿壳内壁，或用 --pins 指定针齿数。")
        curve = _fill(best["theta"], best["rmin"], best["valid"])
        bore_level = float(np.percentile(curve, 20))
        valley = curve < bore_level + 0.25 * max(best["span"] * 0.5, 0.3)
        th = best["theta"][valley]
        bore_pts = np.column_stack([curve[valley] * np.cos(th), curve[valley] * np.sin(th)])
        if len(bore_pts) < 30:
            break
        shift, _ = fit_circle(bore_pts)
        center = center + shift
        chosen = best
        if np.linalg.norm(shift) < 1e-4:
            break
    if chosen is None:
        raise RuntimeError("无法定位针齿节圆中心")

    rel = xy - center
    theta, rmin, valid = extract_edge(rel, chosen["lo"], chosen["hi"], n_bins)
    curve = _fill(theta, rmin, valid)
    pins_n = int(chosen["pins"] if pins is None else pins)
    phase = fourier_phase(theta, curve, pins_n)
    # 波峰不在 0° 就转过去。若转到了齿谷，再错开半个齿距。
    rel = rotate(rel, -phase)
    theta, rmin, valid = extract_edge(rel, chosen["lo"], chosen["hi"], n_bins)
    curve = _fill(theta, rmin, valid)
    pitch = 2 * np.pi / pins_n
    if float(np.interp(0.0, theta, curve)) < float(np.interp(pitch / 2, theta, curve)):
        rel = rotate(rel, -pitch / 2)
        theta, rmin, valid = extract_edge(rel, chosen["lo"], chosen["hi"], n_bins)
        curve = _fill(theta, rmin, valid)
    # 齿槽底部再定一次圆心，去掉整圈的偏心，避免把偏心算进各针齿的跳动。
    for _ in range(3):
        peaks = peak_radii(theta, curve, valid, pins_n)
        finite = np.isfinite(peaks)
        if int(finite.sum()) < max(8, int(pins_n * 0.7)):
            break
        ang = np.arange(pins_n) * pitch
        design = np.column_stack([np.ones(int(finite.sum())), np.cos(ang[finite]), np.sin(ang[finite])])
        coef, *_ = np.linalg.lstsq(design, peaks[finite], rcond=None)
        shift = np.array([float(coef[1]), float(coef[2])])
        if np.linalg.norm(shift) < 1e-4:
            break
        rel = rel - shift
        theta, rmin, valid = extract_edge(rel, chosen["lo"], chosen["hi"], n_bins)
        curve = _fill(theta, rmin, valid)
        phase = fourier_phase(theta, curve, pins_n)
        if abs(phase) > 1e-4:
            rel = rotate(rel, -phase)
            theta, rmin, valid = extract_edge(rel, chosen["lo"], chosen["hi"], n_bins)
            curve = _fill(theta, rmin, valid)
    edge = np.column_stack([curve * np.cos(theta), curve * np.sin(theta)])
    return {
        "xy": rel,
        "theta": theta,
        "radius": curve,
        "valid": valid,
        "edge": edge,
        "pins": pins_n,
        "lo": chosen["lo"],
        "hi": chosen["hi"],
        "center": center,
    }


def peak_radii(theta: np.ndarray, radius: np.ndarray, valid: np.ndarray, pins: int) -> np.ndarray:
    """取每颗针齿方位附近的内轮廓半径，作为该齿外侧半径的初值。"""
    smooth = circular_median(radius, 3)
    peaks = np.full(pins, np.nan)
    pitch = 2 * np.pi / pins
    for i in range(pins):
        alpha = i * pitch
        delta = np.abs(wrap(theta - alpha))
        sel = (delta < np.deg2rad(0.7)) & valid
        if int(sel.sum()) == 0:
            nearby = delta < np.deg2rad(1.4)
            if int(nearby.sum()) == 0:
                continue
            peaks[i] = float(np.median(smooth[nearby]))
        else:
            peaks[i] = float(np.median(smooth[sel]))
    return peaks


def groove_mask(
    theta: np.ndarray,
    radius: np.ndarray,
    valid: np.ndarray,
    pins: int,
    pitch_radius: float,
    pin_radius: float,
    bore: float,
) -> np.ndarray:
    """针齿槽两侧边缘：极角在 0.9×arcsin(rz/Rz) 以内，半径从孔壁外侧直到齿槽。

    孤立外跳点已在 _drop_outer_spikes 里从 valid 去掉，这里不再按齿底收窄。
    """
    limit = half_span(pitch_radius, pin_radius) * 0.9
    delta = wrap(theta - pin_index(theta, pins) * (2 * np.pi / pins))
    low = float(bore) + 0.05
    high = float(pitch_radius) + float(pin_radius) + 0.8
    return valid & (np.abs(delta) <= limit) & (radius >= low) & (radius <= high)


# ---------------------------------------------------------------------------
# 拟合节圆和针齿圆
# ---------------------------------------------------------------------------

def _fit_center_on_ray(pts: np.ndarray, alpha: float, center_radius: float, pin_radius: float) -> float:
    """针齿半径固定，圆心只沿径向移动。用整段齿槽边缘，不只用齿底。"""
    direction = np.array([np.cos(alpha), np.sin(alpha)])
    center_r = float(center_radius)
    for _ in range(12):
        dvec = pts - center_r * direction
        dist = np.maximum(np.hypot(dvec[:, 0], dvec[:, 1]), 1e-9)
        residual = dist - pin_radius
        jac = -(dvec @ direction) / dist
        weight = huber_weight(residual)
        denom = float(np.dot(jac * weight, jac * weight)) + 1e-12
        step = float(np.clip(-np.dot(jac * weight, residual * weight) / denom, -0.25, 0.25))
        center_r += step
        if abs(step) < 1e-5:
            break
    return center_r


def _fit_pin_on_ray(pts: np.ndarray, alpha: float, center_radius: float, pin_radius: float) -> tuple[float, float]:
    """圆心只能沿这颗针齿的径向移动，针齿半径单独自由。"""
    direction = np.array([np.cos(alpha), np.sin(alpha)])
    center_r = float(center_radius)
    radius = float(max(pin_radius, 0.2))
    for _ in range(15):
        dvec = pts - center_r * direction
        dist = np.maximum(np.hypot(dvec[:, 0], dvec[:, 1]), 1e-9)
        residual = dist - radius
        jc = -(dvec @ direction) / dist
        weight = huber_weight(residual)
        design = np.column_stack([jc * weight, -weight])
        step, *_ = np.linalg.lstsq(design, -residual * weight, rcond=None)
        dc = float(np.clip(step[0], -0.25, 0.25))
        dr = float(np.clip(step[1], -0.25, 0.25))
        center_r += dc
        radius = max(0.2, radius + dr)
        if abs(dc) + abs(dr) < 1e-5:
            break
    return center_r, radius


def _residuals(pts: np.ndarray, idx: np.ndarray, offset: np.ndarray, pins: int, pitch_radius: float, pin_radius: float) -> np.ndarray:
    """轮廓点到对应针齿圆的径向差。正值表示点在圆的外面。"""
    pitch = 2 * np.pi / pins
    alpha = idx.astype(float) * pitch
    direction = np.column_stack([np.cos(alpha), np.sin(alpha)])
    center = (pitch_radius + offset[idx])[:, None] * direction
    return np.hypot(pts[:, 0] - center[:, 0], pts[:, 1] - center[:, 1]) - pin_radius


def _bore_radius(theta: np.ndarray, radius: np.ndarray, valid: np.ndarray) -> float:
    """针齿之间那段内孔圆弧的半径。点已经把这段圆弧的圆心放在原点附近。"""
    base = float(np.percentile(radius[valid], 25))
    valley = valid & (radius < base + 0.2)
    pts = np.column_stack([radius[valley] * np.cos(theta[valley]), radius[valley] * np.sin(theta[valley])])
    if len(pts) < 30:
        return base
    center, fitted = fit_circle(pts)
    if np.linalg.norm(center) > 0.3:
        return base
    return float(fitted)


# 圆必须留在外壁的空腔一侧。留出的间隙让整图上圆和外壁不会贴成一条线。
_WALL_CLEARANCE_MM = 0.15
_WALL_COLOR = "#9a3412"
_PIN_CIRCLE_COLOR = "#5b87d6"


def _outer_wall_flags(radius: np.ndarray, theta: np.ndarray, alpha: float) -> np.ndarray:
    """齿槽边缘上靠外的一侧：针齿圆不能进入的外壁。

    比左右邻点突然凹进 0.30 mm 以上的孤立点是飞点，不算外壁。
    """
    radius = np.asarray(radius, dtype=float)
    if len(radius) < 4:
        return np.zeros(len(radius), dtype=bool)
    outer = radius >= np.percentile(radius, 45)
    dang = wrap(theta - alpha)
    flags = outer.copy()
    index = np.arange(len(radius))
    for i in range(len(radius)):
        if not flags[i]:
            continue
        near = (np.abs(wrap(dang - dang[i])) <= np.deg2rad(0.9)) & (index != i)
        if int(near.sum()) >= 2 and radius[i] < float(np.median(radius[near])) - 0.30:
            flags[i] = False
    if int(flags.sum()) < 3:
        return outer
    return flags


def _center_off_wall(
    pts: np.ndarray,
    wall: np.ndarray,
    alpha: float,
    c0: float,
    pin_radius: float,
    clearance: float,
) -> float | None:
    """圆心只沿径向移动。外壁点都留在圆外，齿侧尽量贴合。"""
    if not np.any(wall):
        return float(c0)
    direction = np.array([np.cos(alpha), np.sin(alpha)])
    need = float(pin_radius) + float(clearance)
    flank = ~wall
    best: tuple[tuple[float, float], float] | None = None
    for center_r in np.linspace(float(c0) - 1.2, float(c0) + 0.45, 331):
        dist = np.hypot(pts[:, 0] - center_r * direction[0], pts[:, 1] - center_r * direction[1])
        if float(dist[wall].min()) < need - 1e-9:
            continue
        target = pts[flank] if np.any(flank) else pts[wall]
        miss = np.hypot(target[:, 0] - center_r * direction[0], target[:, 1] - center_r * direction[1])
        score = (float(np.sum((miss - pin_radius) ** 2)), abs(float(center_r) - float(c0)))
        if best is None or score < best[0]:
            best = (score, float(center_r))
    return None if best is None else best[1]


def _enforce_wall_clearance(
    theta: np.ndarray,
    radius: np.ndarray,
    valid: np.ndarray,
    pins: int,
    centers: np.ndarray,
    pin_radius: float,
    bore: float,
    pitch_radius: float,
) -> dict:
    """不干涉：每个外壁点到针齿圆心的距离都不小于针齿半径。

    只在圆确实切入外壁时才移动圆心。噪声级别的正负偏差保持原拟合，
    避免把节圆半径和针齿半径重新对调。
    """
    pitch = 2 * np.pi / pins
    mask = groove_mask(theta, radius, valid, pins, pitch_radius, pin_radius, bore)
    idx = pin_index(theta, pins)
    pockets: list[tuple[np.ndarray, np.ndarray] | None] = []
    wall_chunks: list[np.ndarray] = []
    wall_ids: list[np.ndarray] = []
    gaps = []
    n_cut = 0
    for i in range(pins):
        mine = mask & (idx == i)
        if int(mine.sum()) < 4 or not np.isfinite(centers[i]):
            pockets.append(None)
            continue
        r_i = radius[mine]
        th_i = theta[mine]
        pts = np.column_stack([r_i * np.cos(th_i), r_i * np.sin(th_i)])
        wall = _outer_wall_flags(r_i, th_i, i * pitch)
        pockets.append((pts, wall))
        if not np.any(wall):
            continue
        direction = np.array([np.cos(i * pitch), np.sin(i * pitch)])
        dist = np.hypot(pts[wall, 0] - centers[i] * direction[0], pts[wall, 1] - centers[i] * direction[1])
        gap = float(dist.min()) - float(pin_radius)
        gaps.append(gap)
        if gap < -1e-4:
            n_cut += 1
        wall_chunks.append(pts[wall])
        wall_ids.append(np.full(int(wall.sum()), i, dtype=int))

    new_centers = np.array(centers, dtype=float, copy=True)
    new_radius = float(pin_radius)
    worst = float(min(gaps)) if gaps else float("nan")
    # 切入超过噪声才收。收的时候每一颗都留出同样的可见间隙。
    if gaps and worst < -0.03:
        def place(radius_try: float) -> np.ndarray | None:
            placed = np.array(centers, dtype=float, copy=True)
            for i, pocket in enumerate(pockets):
                if pocket is None:
                    continue
                pts, wall = pocket
                got = _center_off_wall(pts, wall, i * pitch, float(centers[i]), radius_try, _WALL_CLEARANCE_MM)
                if got is None or not np.isfinite(got):
                    return None
                placed[i] = got
            return placed

        placed = place(new_radius)
        if placed is None:
            lo, hi = 0.3, new_radius
            best_c = None
            best_r = None
            for _ in range(18):
                mid = 0.5 * (lo + hi)
                trial = place(mid)
                if trial is None:
                    hi = mid
                else:
                    lo = mid
                    best_c, best_r = trial, mid
            if best_c is None:
                raise RuntimeError("无法在不进入外壁的前提下放下针齿圆")
            new_centers, new_radius = best_c, float(best_r)
        else:
            new_centers = placed

    clearances = []
    for i, pocket in enumerate(pockets):
        if pocket is None:
            continue
        pts, wall = pocket
        if not np.any(wall) or not np.isfinite(new_centers[i]):
            continue
        direction = np.array([np.cos(i * pitch), np.sin(i * pitch)])
        dist = np.hypot(
            pts[wall, 0] - new_centers[i] * direction[0],
            pts[wall, 1] - new_centers[i] * direction[1],
        )
        clearances.append(float(dist.min()) - new_radius)
    return {
        "centers": new_centers,
        "pin_radius": new_radius,
        "pins_cutting_wall": int(n_cut),
        "wall_clearance": float(min(clearances)) if clearances else float("nan"),
        "wall_points": np.vstack(wall_chunks) if wall_chunks else np.zeros((0, 2)),
        "wall_pin": np.concatenate(wall_ids) if wall_ids else np.zeros(0, dtype=int),
    }


def fit_pins(profile: dict) -> dict:
    """每颗针齿用齿槽两侧边缘拟合一个圆。

    圆心限制在该齿的径向上，半径先逐齿自由，针齿半径取中位数。短圆弧上
    圆心和半径会对调，所以再用这个统一半径，沿径向重拟合各圆心。
    节圆半径是这些圆心半径的平均。齿底和孔壁只用来给出初值。
    整圈偏心已在提取轮廓时去掉，这里不再把一次谐波算进半径。
    实测结果还会再沿径向收一次，使圆不进入齿槽外壁。
    """
    pins = int(profile["pins"])
    theta, radius, valid = profile["theta"], profile["radius"], profile["valid"]
    peaks = peak_radii(theta, radius, valid, pins)
    if np.isfinite(peaks).sum() < max(3, pins // 2):
        raise RuntimeError("能拟合的针齿太少，点云可能没有覆盖整圈齿廓")
    bore = _bore_radius(theta, radius, valid)
    pin_radius = float(np.nanmedian(peaks) - bore)
    if pin_radius <= 0.2:
        raise RuntimeError(f"拟合到的针齿半径 {pin_radius:.3f} mm 不合理")
    center_radius = np.where(np.isfinite(peaks), peaks - pin_radius, np.nan)
    finite = np.isfinite(center_radius)
    pitch_radius = float(np.mean(center_radius[finite]))
    pin_radii = np.full(pins, pin_radius)

    for _ in range(3):
        mask = groove_mask(theta, radius, valid, pins, pitch_radius, pin_radius, bore)
        idx = pin_index(theta, pins)
        new_center = center_radius.copy()
        new_radius = pin_radii.copy()
        for i in range(pins):
            mine = mask & (idx == i)
            if int(mine.sum()) < 6:
                continue
            pts = np.column_stack([
                radius[mine] * np.cos(theta[mine]),
                radius[mine] * np.sin(theta[mine]),
            ])
            c0 = float(center_radius[i]) if np.isfinite(center_radius[i]) else pitch_radius
            r0 = float(pin_radii[i]) if np.isfinite(pin_radii[i]) else pin_radius
            center_i, radius_i = _fit_pin_on_ray(pts, i * 2 * np.pi / pins, c0, r0)
            if not np.isfinite(center_i) or not np.isfinite(radius_i):
                continue
            if not 0.4 <= radius_i <= 8.0 or abs(center_i - pitch_radius) > 2.0:
                continue
            new_center[i] = center_i
            new_radius[i] = radius_i
        good = np.isfinite(new_center) & np.isfinite(new_radius)
        if int(good.sum()) < max(3, pins // 2):
            break
        center_radius = new_center
        pin_radii = new_radius
        pitch_radius = float(np.mean(center_radius[good]))
        pin_radius = float(np.median(pin_radii[good]))
        if pin_radius <= 0.2:
            raise RuntimeError(f"拟合到的针齿半径 {pin_radius:.3f} mm 不合理")

    finite = np.isfinite(pin_radii)
    if int(finite.sum()) < max(3, pins // 2):
        raise RuntimeError("能拟合的针齿太少，点云可能没有覆盖整圈齿廓")
    pin_radius = float(np.median(pin_radii[finite]))
    free_center = center_radius.copy()
    if pin_radius <= 0.2:
        raise RuntimeError(f"拟合到的针齿半径 {pin_radius:.3f} mm 不合理")
    mask = groove_mask(theta, radius, valid, pins, pitch_radius, pin_radius, bore)
    idx = pin_index(theta, pins)
    fitted_center = np.full(pins, np.nan)
    for i in range(pins):
        mine = mask & (idx == i)
        if int(mine.sum()) < 6:
            continue
        pts_i = np.column_stack([
            radius[mine] * np.cos(theta[mine]),
            radius[mine] * np.sin(theta[mine]),
        ])
        c0 = float(center_radius[i]) if np.isfinite(center_radius[i]) else pitch_radius
        fitted_center[i] = _fit_center_on_ray(pts_i, i * 2 * np.pi / pins, c0, pin_radius)
    good = np.isfinite(fitted_center)
    if int(good.sum()) < max(3, pins // 2):
        raise RuntimeError("能拟合的针齿太少，点云可能没有覆盖整圈齿廓")
    pitch_radius = float(np.mean(fitted_center[good]))
    offset = np.zeros(pins)
    offset[good] = fitted_center[good] - pitch_radius

    mask = groove_mask(theta, radius, valid, pins, pitch_radius, pin_radius, bore)
    idx = pin_index(theta, pins)
    pts = np.column_stack([
        radius[mask] * np.cos(theta[mask]),
        radius[mask] * np.sin(theta[mask]),
    ]) if int(mask.sum()) else np.zeros((0, 2))
    arc_index = idx[mask]
    form = _residuals(pts, arc_index, offset, pins, pitch_radius, pin_radius) if len(pts) else np.zeros(0)
    edge_count = np.array([int((mask & (idx == i)).sum()) for i in range(pins)])
    return {
        "pitch_radius": float(pitch_radius),
        "pin_radius": float(pin_radius),
        "pin_radii": pin_radii,
        "free_center": free_center,
        "offset": offset,
        "bore_radius": bore,
        "peaks": peaks,
        "arc_points": pts,
        "arc_index": arc_index,
        "form": form,
        "edge_count": edge_count,
        "half_angle_deg": float(np.degrees(half_span(pitch_radius, pin_radius) * 0.9)),
        "pins": pins,
    }


def evaluate(profile: dict, fit: dict, design: dict | None) -> dict:
    """相对设计圆（或拟合圆）计算每个轮廓点、每颗针齿的偏差。"""
    pins = int(fit["pins"])
    pitch = 2 * np.pi / pins
    theta, radius, valid = profile["theta"], profile["radius"], profile["valid"]
    ref_pitch = design["pitch_radius"] if design else fit["pitch_radius"]
    ref_pin = design["pin_radius"] if design else fit["pin_radius"]
    idx = pin_index(theta, pins)
    alpha = idx.astype(float) * pitch
    direction = np.column_stack([np.cos(alpha), np.sin(alpha)])
    points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    delta = wrap(theta - alpha)
    on_arc = groove_mask(
        theta, radius, valid, pins, fit["pitch_radius"], fit["pin_radius"], fit["bore_radius"],
    )
    # 齿间圆弧只取两颗针齿正中间的那一段，不把齿侧算进去。
    bore = valid & (np.abs(delta) > pitch * 0.36) & (radius < ref_pitch + 0.6) & ~on_arc

    fit_center = (fit["pitch_radius"] + fit["offset"][idx])[:, None] * direction
    dist_fit = np.hypot(points[:, 0] - fit_center[:, 0], points[:, 1] - fit_center[:, 1])
    own_radius = np.asarray(fit.get("pin_radii", np.full(pins, fit["pin_radius"])), dtype=float)
    own_radius = np.where(np.isfinite(own_radius), own_radius, fit["pin_radius"])
    profile_error = np.full(len(theta), np.nan)
    own_for_error = own_radius[idx] if fit.get("per_pin_radius") else np.full(len(theta), fit["pin_radius"])
    profile_error[on_arc] = dist_fit[on_arc] - own_for_error[on_arc]
    free_center = np.asarray(fit.get("free_center", fit["pitch_radius"] + fit["offset"]), dtype=float)
    stable = fit["pitch_radius"] + fit["offset"]
    free_center = np.where(np.isfinite(free_center), free_center, stable)
    dist_free = np.hypot(points[:, 0] - free_center[idx] * direction[:, 0], points[:, 1] - free_center[idx] * direction[:, 1])
    form_error = np.full(len(theta), np.nan)
    form_error[on_arc] = dist_free[on_arc] - own_radius[idx][on_arc]

    per_pin = []
    center_radius = fit["pitch_radius"] + fit["offset"]
    for i in range(pins):
        arc_i = on_arc & (idx == i)
        form_i = arc_i
        per_pin.append({
            "angle_deg": float(np.degrees(i * pitch)),
            "center_radius": float(center_radius[i]),
            "pitch_error": float(center_radius[i] - ref_pitch),
            "runout": float(fit["offset"][i]),
            "profile": stats(profile_error[arc_i]),
            "form": stats(form_error[form_i]),
            "n": int(arc_i.sum()),
            "peak_radius": float(fit["peaks"][i]) if np.isfinite(fit["peaks"][i]) else float("nan"),
            "diameter": float(2.0 * own_radius[i]),
        })
    return {
        "profile_error": profile_error,
        "form_error": form_error,
        "on_arc": on_arc,
        "bore": bore,
        "per_pin": per_pin,
        "profile_stats": stats(profile_error[on_arc]),
        "form_stats": stats(form_error[on_arc]),
        "bore_stats": stats(radius[bore] - ref_pitch),
        "ref_pitch": ref_pitch,
        "ref_pin": ref_pin,
    }


def _apply_wall_clearance(profile: dict, fit: dict) -> dict:
    """每颗针齿用自己的齿槽边缘定直径。切入外壁的，只缩小这一颗的半径。

    圆心仍在该齿径向上。噪声级别的贴边不改共用半径拟合，避免自检被飞点带走。
    设计模型不走这里。
    """
    pins = int(fit["pins"])
    pitch = 2 * np.pi / pins
    radii = np.asarray(fit["pin_radii"], dtype=float).copy()
    centers = np.asarray(fit["free_center"], dtype=float).copy()
    locked = np.asarray(fit["pitch_radius"], dtype=float) + np.asarray(fit["offset"], dtype=float)
    radii = np.where(np.isfinite(radii), radii, float(fit["pin_radius"]))
    centers = np.where(np.isfinite(centers), centers, locked)
    theta, radius, valid = profile["theta"], profile["radius"], profile["valid"]
    med = float(np.median(radii))
    pitch0 = float(np.mean(centers))
    mask = groove_mask(theta, radius, valid, pins, pitch0, med, float(fit["bore_radius"]))
    idx = pin_index(theta, pins)
    n_cut = 0
    deep = False
    wall_chunks: list[np.ndarray] = []
    wall_ids: list[np.ndarray] = []
    clearances: list[float] = []
    for i in range(pins):
        mine = mask & (idx == i)
        if int(mine.sum()) < 4 or not np.isfinite(centers[i]):
            continue
        r_i = radius[mine]
        th_i = theta[mine]
        pts = np.column_stack([r_i * np.cos(th_i), r_i * np.sin(th_i)])
        wall = _outer_wall_flags(r_i, th_i, i * pitch)
        if not np.any(wall):
            continue
        direction = np.array([np.cos(i * pitch), np.sin(i * pitch)])
        dist = np.hypot(pts[wall, 0] - centers[i] * direction[0], pts[wall, 1] - centers[i] * direction[1])
        limit = float(dist.min())
        wall_chunks.append(pts[wall])
        wall_ids.append(np.full(int(wall.sum()), i, dtype=int))
        if limit < float(radii[i]) - 1e-4:
            n_cut += 1
            if limit < float(radii[i]) - 0.03:
                deep = True
            radii[i] = max(0.2, limit - 0.01)
        clearances.append(limit - float(radii[i]))

    wall_points = np.vstack(wall_chunks) if wall_chunks else np.zeros((0, 2))
    wall_pin = np.concatenate(wall_ids) if wall_ids else np.zeros(0, dtype=int)
    clearance = float(min(clearances)) if clearances else float("nan")
    if not deep:
        updated = dict(fit)
        updated["pins_cutting_wall"] = int(n_cut)
        updated["wall_clearance"] = clearance
        updated["wall_points"] = wall_points
        updated["wall_pin"] = wall_pin
        updated["per_pin_radius"] = False
        return updated

    pitch_radius = float(np.mean(centers))
    offset = centers - pitch_radius
    pin_radius = float(np.median(radii))
    updated = dict(fit)
    updated["pitch_radius"] = pitch_radius
    updated["pin_radius"] = pin_radius
    updated["pin_radii"] = radii
    updated["free_center"] = centers.copy()
    updated["offset"] = offset
    updated["per_pin_radius"] = True
    updated["pins_cutting_wall"] = int(n_cut)
    updated["wall_clearance"] = clearance
    updated["wall_points"] = wall_points
    updated["wall_pin"] = wall_pin
    mask = groove_mask(theta, radius, valid, pins, pitch_radius, pin_radius, float(fit["bore_radius"]))
    idx = pin_index(theta, pins)
    pts = np.column_stack([
        radius[mask] * np.cos(theta[mask]),
        radius[mask] * np.sin(theta[mask]),
    ]) if int(mask.sum()) else np.zeros((0, 2))
    updated["arc_points"] = pts
    updated["arc_index"] = idx[mask]
    updated["form"] = (
        _residuals(pts, idx[mask], offset, pins, pitch_radius, pin_radius) if len(pts) else np.zeros(0)
    )
    updated["edge_count"] = np.array([int((mask & (idx == i)).sum()) for i in range(pins)])
    updated["half_angle_deg"] = float(np.degrees(half_span(pitch_radius, float(np.max(radii))) * 0.9))
    return updated


def analyze_xy(xy: np.ndarray, design: dict | None, n_bins: int, pins: int | None) -> dict:
    """从平面点云提取针齿廓，拟合圆心和半径，并计算齿廓偏差。"""
    forced = pins if pins is not None else (design["pins"] if design else None)
    profile = select_profile(xy, n_bins, pins=forced)
    profile["alignment"] = {"dx": 0.0, "dy": 0.0, "dphase_deg": 0.0}
    fit = _apply_wall_clearance(profile, fit_pins(profile))
    evaluation = evaluate(profile, fit, design)
    return {"profile": profile, "fit": fit, "eval": evaluation, "design": design}


# ---------------------------------------------------------------------------
# 设计值：模型或参数
# ---------------------------------------------------------------------------

def _mesh_frame(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用模型顶点拟合轴线方向和两个互相垂直的平面方向。"""
    verts = triangles.reshape(-1, 3)
    origin = verts.mean(axis=0)
    _, _, vt = np.linalg.svd(verts - origin, full_matrices=False)
    axis = vt[2].copy()
    e1 = vt[0].copy()
    e2 = np.cross(axis, e1)
    e2 /= np.linalg.norm(e2)
    return origin, axis, e1, e2


def _section(triangles: np.ndarray, origin: np.ndarray, axis: np.ndarray, e1: np.ndarray, e2: np.ndarray, height: float) -> np.ndarray:
    """用垂直于轴线的平面去切三角形网格，得到该高度上的截面点。"""
    hits = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        h0 = (triangles[:, a] - origin) @ axis
        h1 = (triangles[:, b] - origin) @ axis
        cross = (h0 - height) * (h1 - height) < 0
        if not np.any(cross):
            continue
        t = (height - h0[cross]) / (h1[cross] - h0[cross])
        point = triangles[cross, a] + t[:, None] * (triangles[cross, b] - triangles[cross, a])
        rel = point - origin
        hits.append(np.column_stack([rel @ e1, rel @ e2]))
    if not hits:
        return np.zeros((0, 2))
    return np.vstack(hits)


def _scale_mesh(triangles: np.ndarray, scan_radius: float) -> tuple[np.ndarray, str]:
    """按测量件的半径判断模型单位，并把模型换算成毫米。"""
    origin, axis, e1, e2 = _mesh_frame(triangles)
    rel = triangles.reshape(-1, 3) - origin
    radius = np.hypot(rel @ e1, rel @ e2)
    model_radius = float(np.percentile(radius, 90))
    if model_radius <= 0:
        return triangles, "未换算"
    ratio = scan_radius / model_radius
    if 800 <= ratio <= 1200:
        return triangles * 1000.0, "模型数值约为米，已乘 1000 换成毫米"
    if 80 <= ratio <= 120:
        return triangles * 100.0, "模型数值约为分米，已乘 100 换成毫米"
    if 8 <= ratio <= 12:
        return triangles * 10.0, "模型数值约为厘米，已乘 10 换成毫米"
    if not 0.4 <= ratio <= 2.5:
        raise RuntimeError(
            f"模型和测量的尺寸差太多（测量半径约 {scan_radius:.2f}，模型半径约 {model_radius:.4f}）。"
            "请确认单位，或先把模型换成毫米。"
        )
    return triangles, "模型按毫米使用"


def design_from_mesh(triangles: np.ndarray, scan_radius: float, n_bins: int) -> tuple[dict, str]:
    """在模型的若干截面里找出针齿廓，得到针齿数、节圆半径和针齿半径。"""
    triangles, unit_note = _scale_mesh(triangles, scan_radius)
    origin, axis, e1, e2 = _mesh_frame(triangles)
    height = (triangles.reshape(-1, 3) - origin) @ axis
    best = None
    for z0 in np.linspace(float(np.percentile(height, 3)), float(np.percentile(height, 97)), 36):
        xy = _section(triangles, origin, axis, e1, e2, float(z0))
        if len(xy) < 200:
            continue
        try:
            profile = select_profile(xy, n_bins)
            fit = fit_pins(profile)
        except RuntimeError:
            continue
        form = robust_rms(fit["form"]) if fit["form"].size else float("inf")
        spread = float(np.ptp(fit["offset"])) if np.isfinite(fit["offset"]).all() else float("inf")
        # 针齿是同一节圆上的一圈圆弧。螺栓孔之类的圆不满足这一点。
        if form > 0.08 or spread > 0.2 or fit["pins"] < 8:
            continue
        harmonic, _, _ = _pin_score(profile["radius"], profile["valid"])
        rank = (fit["pins"], harmonic)
        if best is None or rank > best[0]:
            best = (rank, z0, fit, profile)
    if best is None:
        raise RuntimeError(
            "三维模型里没有找到一整圈针齿圆弧。"
            "请确认模型是针齿壳，或改用 --pins --pitch-diameter --pin-diameter。"
        )
    _, z0, fit, profile = best
    pitch_radius, pin_radius = _design_circles(profile, fit)
    design = {
        "pins": int(fit["pins"]),
        "pitch_radius": pitch_radius,
        "pin_radius": pin_radius,
        "source": (
            f"三维模型截面（沿轴向高度 {z0:.2f} mm），"
            f"拟合针齿数 {fit['pins']}，节圆半径 {pitch_radius:.4f} mm，"
            f"针齿半径 {pin_radius:.4f} mm。{unit_note}"
        ),
    }
    return design, design["source"]


def _design_circles(profile: dict, fit: dict) -> tuple[float, float]:
    """模型截面是干净圆弧时，直接拟合每颗针齿的圆，得到圆心圆半径和针齿半径。"""
    pts = profile["xy"]
    radius = np.hypot(pts[:, 0], pts[:, 1])
    angle = np.arctan2(pts[:, 1], pts[:, 0])
    pins = int(fit["pins"])
    pitch = 2 * np.pi / pins
    limit = half_span(fit["pitch_radius"], fit["pin_radius"]) * 1.15
    low = fit["bore_radius"] - 0.4
    high = fit["pitch_radius"] + fit["pin_radius"] + 0.6
    centers = []
    radii = []
    for i in range(pins):
        alpha = i * pitch
        picked = (np.abs(wrap(angle - alpha)) <= limit) & (radius >= low) & (radius <= high)
        mine = pts[picked]
        if len(mine) < 8:
            continue
        center, rad = fit_circle(mine)
        centers.append(float(np.hypot(center[0], center[1])))
        radii.append(rad)
    if len(centers) < pins * 0.8:
        return fit["pitch_radius"], fit["pin_radius"]
    centers = np.asarray(centers)
    radii = np.asarray(radii)
    if float(np.std(centers)) > 0.05 or float(np.std(radii)) > 0.05:
        return fit["pitch_radius"], fit["pin_radius"]
    return float(np.mean(centers)), float(np.mean(radii))


def design_from_parameters(pins: int, pitch_diameter: float, pin_diameter: float) -> dict:
    """用针齿数、节圆直径和针齿直径组成设计值。"""
    if pins < 8:
        raise ValueError("针齿数至少为 8")
    pitch_radius = pitch_diameter / 2.0
    pin_radius = pin_diameter / 2.0
    if min(pitch_radius, pin_radius) <= 0:
        raise ValueError("节圆直径和针齿直径都必须大于 0")
    if pin_radius >= pitch_radius:
        raise ValueError("针齿直径必须小于节圆直径")
    return {
        "pins": pins,
        "pitch_radius": pitch_radius,
        "pin_radius": pin_radius,
        "source": (
            f"设计参数：针齿数 {pins}，节圆直径 {pitch_diameter:g} mm，针齿直径 {pin_diameter:g} mm"
        ),
    }


def rough_radius(points: np.ndarray) -> float:
    """估计点云在盘面上的 90 分位半径，用来和模型尺寸对照。"""
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    normal = vt[2]
    e1 = vt[0]
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)
    rel = points - center
    radius = np.hypot(rel @ e1, rel @ e2)
    return float(np.percentile(radius, 90))


def project_scan(points: np.ndarray) -> np.ndarray:
    """拟合测量平面，返回平面坐标。扫描件通常是针齿那一面。"""
    mask = np.ones(len(points), dtype=bool)
    center = points.mean(axis=0)
    normal = np.array([0.0, 0.0, 1.0])
    vt = None
    for _ in range(6):
        pts = points[mask]
        center = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
        normal = vt[2].copy()
        if normal[2] < 0:
            normal = -normal
        height = (points - center) @ normal
        med = np.median(height[mask])
        mad = np.median(np.abs(height[mask] - med)) + 1e-9
        new_mask = np.abs(height - med) < max(0.4, 8 * mad)
        if new_mask.sum() < max(100, int(0.2 * len(points))):
            break
        mask = new_mask
    e1 = vt[0].copy()
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)
    rel = points - center
    return np.column_stack([rel @ e1, rel @ e2])


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _fmt(s: dict) -> str:
    """把一组偏差统计格式化成一行文字。"""
    if not np.isfinite(s["mean"]):
        return "无数据"
    return (
        f"平均 {s['mean']:+.4f} mm，标准差 {s['std']:.4f} mm，RMS {s['rms']:.4f} mm，"
        f"最小 {s['min']:+.4f} mm，最大 {s['max']:+.4f} mm，峰谷 {s['pv']:.4f} mm"
    )


def write_report(out: Path, payload: dict) -> None:
    """写出针齿齿廓误差的文字摘要和逐齿 CSV。"""
    out.mkdir(parents=True, exist_ok=True)
    fit = payload["fit"]
    evaluation = payload["eval"]
    design = payload["design"]
    pins = fit["pins"]
    lines = [
        "针齿齿廓与节圆半径误差",
        "",
        f"测量文件：{payload['scan']}",
        f"设计：{design['source'] if design else '未提供。下面以测量点云自己拟合的节圆为基准，偏差表示各针齿相对这条节圆的不一致'}",
        f"测量点数：{payload['n_points']}",
        f"针齿数：{pins}",
        "坐标：节圆中心在原点，1 号针齿在 0°，其余按逆时针编号。",
        "",
        "节圆（针齿中心圆）",
        f"实测半径 {fit['pitch_radius']:.4f} mm",
    ]
    if design:
        err = fit["pitch_radius"] - design["pitch_radius"]
        lines.append(f"设计半径 {design['pitch_radius']:.4f} mm")
        lines.append(f"节圆半径误差 {err:+.4f} mm（正值 = 针齿中心整体偏外）")
    runout = np.array([row["runout"] for row in evaluation["per_pin"]])
    lines.append("各针齿中心相对实测节圆的径向跳动 " + _fmt(stats(runout)))
    if design:
        pitch_err = np.array([row["pitch_error"] for row in evaluation["per_pin"]])
        lines.append("各针齿中心相对设计节圆的径向偏差 " + _fmt(stats(pitch_err)))
    lines += [
        "",
        "针齿半径",
        f"实测半径 {fit['pin_radius']:.4f} mm",
    ]
    if design:
        lines.append(f"设计半径 {design['pin_radius']:.4f} mm")
        lines.append(f"针齿半径误差 {fit['pin_radius'] - design['pin_radius']:+.4f} mm（正值 = 齿槽圆弧偏大）")
    if "wall_clearance" in fit and np.isfinite(fit["wall_clearance"]):
        lines.append(
            f"外壁：调整前 {int(fit['pins_cutting_wall'])} 颗针齿的圆切入外壁。"
            f"现圆在空腔一侧，与外壁最小间隙 {fit['wall_clearance']:.4f} mm。"
        )
    if evaluation["per_pin"] and "diameter" in evaluation["per_pin"][0]:
        diameters = [row["diameter"] for row in evaluation["per_pin"]]
        lines.append(
            "各针齿拟合直径 (mm)："
            + "，".join(f"{i + 1}号 {d:.2f}" for i, d in enumerate(diameters))
        )
        lines.append(
            f"直径最小 {min(diameters):.2f} mm，最大 {max(diameters):.2f} mm"
        )
    half_deg = fit.get("half_angle_deg")
    counts = fit.get("edge_count")
    if half_deg is not None and counts is not None:
        use = np.asarray(counts)
        lines += [
            "",
            (
                f"齿槽边缘：极角半宽 ±{half_deg:.2f}°（0.9×arcsin(rz/Rz)），"
                f"每齿 {int(use.min())}–{int(use.max())} 点，中位数 {float(np.median(use)):.0f} 点"
            ),
        ]
    lines += [
        "",
        "齿廓偏差（测点到该针齿拟合圆心的距离 − 拟合针齿半径；含齿槽两侧边缘）",
        "正值 = 测点离圆心比针齿半径更远 = 齿槽偏大",
        _fmt(evaluation["profile_stats"]),
        "",
        "齿廓形状偏差（圆心放在各针齿实测中心上，半径用该针齿自己的拟合半径）",
        _fmt(evaluation["form_stats"]),
        "",
        "齿间圆弧（针齿之间的内孔，相对节圆）",
        f"实测半径 {fit['bore_radius']:.4f} mm",
        _fmt(evaluation["bore_stats"]),
        "",
    ]
    if payload["tolerance"] is not None:
        tol = payload["tolerance"]
        err = evaluation["profile_error"][evaluation["on_arc"]]
        err = err[np.isfinite(err)]
        over = np.abs(err) > tol
        lines.append(f"齿廓公差带 ±{tol:.4f} mm：超差 {int(over.sum())} / {len(err)} 个采样（{100 * over.mean():.2f}%）")
        lines.append("")
    lines.append("各针齿（半径和偏差单位 mm）")
    lines.append("齿号, 中心角°, 中心半径, 相对设计节圆, 相对实测节圆, 齿廓平均, 齿廓最小, 齿廓最大, 齿廓RMS, 形状RMS, 点数")
    for i, row in enumerate(evaluation["per_pin"]):
        p = row["profile"]
        lines.append(
            f"{i + 1}, {row['angle_deg']:.2f}, {row['center_radius']:.4f}, {row['pitch_error']:+.4f}, "
            f"{row['runout']:+.4f}, {p['mean']:+.4f}, {p['min']:+.4f}, {p['max']:+.4f}, {p['rms']:.4f}, "
            f"{row['form']['rms']:.4f}, {row['n']}"
        )
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)

    with (out / "pin_error.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pin", "angle_deg", "center_radius_mm", "pitch_error_mm", "runout_mm",
            "profile_mean_mm", "profile_min_mm", "profile_max_mm", "profile_rms_mm",
            "form_rms_mm", "n",
        ])
        for i, row in enumerate(evaluation["per_pin"]):
            p = row["profile"]
            writer.writerow([
                i + 1, f"{row['angle_deg']:.4f}", f"{row['center_radius']:.5f}",
                f"{row['pitch_error']:.5f}", f"{row['runout']:.5f}",
                f"{p['mean']:.5f}", f"{p['min']:.5f}", f"{p['max']:.5f}", f"{p['rms']:.5f}",
                f"{row['form']['rms']:.5f}", row["n"],
            ])

    prof = payload["profile"]
    err = evaluation["profile_error"]
    with (out / "profile_error.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "x_mm", "y_mm", "radius_mm", "normal_error_mm", "region", "pin"])
        order = np.argsort(prof["theta"])
        for i in order:
            if not np.isfinite(err[i]):
                continue
            region = "arc" if evaluation["on_arc"][i] else "bore"
            writer.writerow([
                f"{np.degrees(prof['theta'][i]):.4f}",
                f"{prof['edge'][i, 0]:.5f}",
                f"{prof['edge'][i, 1]:.5f}",
                f"{prof['radius'][i]:.5f}",
                f"{err[i]:.5f}",
                region,
                int(pin_index(np.array([prof["theta"][i]]), pins)[0]) + 1,
            ])


def _limit(err: np.ndarray) -> float:
    """取偏差绝对值的 98 分位，作为着色的尺度。"""
    use = np.abs(err[np.isfinite(err)])
    if len(use) == 0:
        return 0.1
    return max(float(np.percentile(use, 98)), 1e-4)


# 负到正：靛蓝、蓝、青、青绿、黄、橙、红、酒红。相邻档换色相，零两侧只保留浅色。
_ERROR_BAND = np.array([
    [42, 8, 120],
    [22, 52, 220],
    [0, 132, 235],
    [0, 186, 214],
    [0, 166, 132],
    [34, 186, 62],
    [40, 210, 150],
    [214, 248, 255],
    [255, 228, 186],
    [255, 236, 0],
    [255, 140, 0],
    [255, 51, 0],
    [210, 0, 40],
    [186, 0, 150],
    [120, 36, 0],
    [72, 0, 28],
], dtype=float) / 255.0


def _step_colors(n_bin: int) -> np.ndarray:
    """按偏差档数从色带里取颜色。"""
    if n_bin == len(_ERROR_BAND):
        colors = _ERROR_BAND
    else:
        index = np.linspace(0, len(_ERROR_BAND) - 1, n_bin)
        colors = _ERROR_BAND[np.clip(np.rint(index).astype(int), 0, len(_ERROR_BAND) - 1)]
    return np.column_stack([colors, np.ones(len(colors))])


def _mm_color_scale(values: np.ndarray, step: float = 0.05):
    """以 0 为中心、每 0.05 mm 一档，显示范围固定为 ±0.40 mm。超出的值贴到两端颜色。"""
    del values
    half = 0.40
    n_edge = int(round(2 * half / step)) + 1
    edges = np.linspace(-half, half, n_edge)
    n_bin = len(edges) - 1
    cmap = ListedColormap(_step_colors(n_bin))
    norm = BoundaryNorm(edges, n_bin, clip=True)
    tick_step = step if n_edge <= 21 else step * 2
    ticks = np.linspace(-half, half, int(round(2 * half / tick_step)) + 1)
    return cmap, norm, ticks, half


def _banded_colorbar(fig, mappable, ax, ticks, **kwargs):
    """色条按 0.05 mm 边界画成独立色块，档与档之间不渐变。"""
    cb = fig.colorbar(
        mappable,
        ax=ax,
        ticks=ticks,
        boundaries=np.asarray(mappable.norm.boundaries, dtype=float),
        spacing="uniform",
        drawedges=True,
        **kwargs,
    )
    solids = getattr(cb, "solids", None)
    if solids is not None and hasattr(solids, "set_antialiased"):
        solids.set_antialiased(False)
    if getattr(cb, "dividers", None) is not None:
        cb.dividers.set_color("#202124")
        cb.dividers.set_linewidth(0.8)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    return cb


def _design_curve(pins: int, pitch_radius: float, pin_radius: float, n: int = 80) -> np.ndarray:
    """用节圆半径和针齿半径生成一圈针齿外轮廓。"""
    theta = np.linspace(-np.pi, np.pi, pins * n, endpoint=False)
    radius = np.full(len(theta), pitch_radius)
    direction = np.column_stack([np.cos(theta), np.sin(theta)])
    pitch = 2 * np.pi / pins
    for i in range(pins):
        center = pitch_radius * np.array([np.cos(i * pitch), np.sin(i * pitch)])
        proj = direction @ center
        disc = proj ** 2 - np.dot(center, center) + pin_radius ** 2
        hit = disc > 0
        far = proj.copy()
        far[hit] = proj[hit] + np.sqrt(disc[hit])
        radius = np.maximum(radius, far)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


_FIT_PITCH_COLOR = "#188038"
_DESIGN_PITCH_COLOR = "#d06a00"


def _pitch_pair(payload: dict) -> tuple[float, float] | None:
    """返回（实测节圆半径, 设计节圆半径）。没有设计值时不标注节圆误差。"""
    design = payload.get("design")
    if not design:
        return None
    return float(payload["fit"]["pitch_radius"]), float(design["pitch_radius"])


def _pitch_error_text(measured: float, design: float) -> str:
    """节圆半径误差的标注文字。"""
    return f"节圆误差 {measured - design:+.3f} mm"


def _gap_angle(pins: int) -> float:
    """1 号齿与 2 号齿正中间，避开针齿圆。"""
    return np.pi / pins


def _draw_pitch_circles(ax, measured: float, design: float | None, fitted_lw: float = 1.15) -> None:
    """画实测节圆和设计节圆。"""
    if design is not None:
        ax.add_patch(plt.Circle(
            (0.0, 0.0), design,
            fill=False, ls="-", ec=_DESIGN_PITCH_COLOR, lw=0.95, zorder=4,
        ))
    ax.add_patch(plt.Circle(
        (0.0, 0.0), measured,
        fill=False, ls="--", ec=_FIT_PITCH_COLOR, lw=fitted_lw, zorder=5,
    ))


def _mark_pitch_gap(ax, measured: float, design: float, angle: float) -> None:
    """在两节圆之间画径向尺寸线。间隙只有零点几毫米，放大图里才看得清数字。"""
    radial = np.array([np.cos(angle), np.sin(angle)])
    tangential = np.array([-radial[1], radial[0]])
    extension = 0.7
    inner = min(measured, design) - extension
    outer = max(measured, design) + extension
    ax.plot(
        [inner * radial[0], outer * radial[0]],
        [inner * radial[1], outer * radial[1]],
        color="#202124", lw=0.8, zorder=7, solid_capstyle="butt",
    )
    serif = 1.2
    for radius in (measured, design):
        point = radius * radial
        ax.plot(
            [point[0] - serif * tangential[0], point[0] + serif * tangential[0]],
            [point[1] - serif * tangential[1], point[1] + serif * tangential[1]],
            color="#202124", lw=0.8, zorder=7, solid_capstyle="butt",
        )


def _draw_pitch_zoom(ax, measured: float, design: float, angle: float) -> None:
    """把两节圆的径向间隙放大，标出带符号的节圆误差。"""
    radial = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    phi = np.linspace(-0.03, 0.03, 120)
    styles = (
        (design, _DESIGN_PITCH_COLOR, "-", "设计节圆"),
        (measured, _FIT_PITCH_COLOR, "--", "拟合节圆"),
    )
    for radius, color, style, name in styles:
        arc = np.column_stack([
            radius * np.cos(angle + phi),
            radius * np.sin(angle + phi),
        ])
        ax.plot(arc[:, 0], arc[:, 1], color=color, lw=1.5, ls=style, zorder=3, label=name)
    inner = min(measured, design) * radial
    outer = max(measured, design) * radial
    ax.annotate(
        "",
        xy=(outer[0], outer[1]),
        xytext=(inner[0], inner[1]),
        arrowprops=dict(arrowstyle="<->", color="#202124", lw=1.0, shrinkA=0, shrinkB=0),
        zorder=4,
    )
    gap = abs(measured - design)
    half = max(gap * 4.2, 0.5)
    origin = 0.5 * (measured + design) * radial
    ax.set_xlim(origin[0] - half, origin[0] + half)
    ax.set_ylim(origin[1] - half, origin[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.set_title(_pitch_error_text(measured, design), fontproperties=_cjk_font(8.5), color="#202124", pad=3)
    ax.tick_params(labelsize=6, colors="#5f6368", length=2.2, pad=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for spine in ax.spines.values():
        spine.set_color("#9aa0a6")
        spine.set_linewidth(0.7)
    ax.legend(
        loc="upper left",
        prop=_cjk_font(7.5),
        frameon=True,
        framealpha=0.94,
        borderpad=0.25,
        handlelength=1.6,
        handletextpad=0.4,
    )


def _draw_scan_background(ax, xy: np.ndarray) -> None:
    """把同一坐标系里的原始点云画成浅灰背景。"""
    cloud = np.asarray(xy, dtype=float)
    if len(cloud) == 0:
        return
    ax.patch.set_zorder(0)
    ax.scatter(
        cloud[:, 0],
        cloud[:, 1],
        s=0.35,
        c="#c5c8ce",
        linewidths=0,
        marker=".",
        zorder=1,
        rasterized=True,
    )


def _draw_outer_wall(ax, fit: dict, lw: float = 1.2, zorder: float = 4) -> None:
    """把每颗齿槽的外壁画成独立的折线，和针齿圆分开。"""
    pts = np.asarray(fit.get("wall_points", np.zeros((0, 2))), dtype=float)
    pin = np.asarray(fit.get("wall_pin", np.zeros(0, dtype=int)))
    if len(pts) == 0:
        return
    for i in np.unique(pin):
        mine = pts[pin == i]
        order = np.argsort(np.arctan2(mine[:, 1], mine[:, 0]))
        mine = mine[order]
        ax.plot(
            mine[:, 0], mine[:, 1],
            color=_WALL_COLOR, lw=lw, solid_capstyle="round", zorder=zorder,
        )


def _circle_radii(fit: dict) -> np.ndarray:
    """图上每个针齿圆实际使用的半径。各齿分开拟合时不用同一个数。"""
    pins = int(fit["pins"])
    shared = float(fit["pin_radius"])
    if not fit.get("per_pin_radius"):
        return np.full(pins, shared)
    radii = np.asarray(fit.get("pin_radii", np.full(pins, shared)), dtype=float)
    return np.where(np.isfinite(radii), radii, shared)


def _draw_pin_circles(ax, fit: dict, lw: float = 0.9, zorder: float = 3) -> None:
    """按拟合出的圆心和半径画每一颗针齿圆。"""
    pins = int(fit["pins"])
    radii = _circle_radii(fit)
    center_r = np.asarray(fit["pitch_radius"], dtype=float) + np.asarray(fit["offset"], dtype=float)
    ang = np.arange(pins) * (2.0 * np.pi / pins)
    for i in range(pins):
        ax.add_patch(plt.Circle(
            (float(center_r[i] * np.cos(ang[i])), float(center_r[i] * np.sin(ang[i]))),
            float(radii[i]), fill=False, ec=_PIN_CIRCLE_COLOR, lw=lw, zorder=zorder,
        ))


def _draw_diameter_labels(ax, fit: dict) -> None:
    """把每颗针齿的直径标在齿外侧。"""
    pins = int(fit["pins"])
    radii = _circle_radii(fit)
    center_r = np.asarray(fit["pitch_radius"], dtype=float) + np.asarray(fit["offset"], dtype=float)
    ang = np.arange(pins) * (2.0 * np.pi / pins)
    label_r = float(np.max(center_r + radii) + 7.0)
    ax.scatter(label_r * np.cos(ang), label_r * np.sin(ang), s=0.2, c="white", alpha=0.0, zorder=0)
    font = _cjk_font(6.5)
    for i in range(pins):
        c, s = float(np.cos(ang[i])), float(np.sin(ang[i]))
        ha = "left" if c > 0.2 else "right" if c < -0.2 else "center"
        va = "bottom" if s > 0.2 else "top" if s < -0.2 else "center"
        ax.text(
            label_r * c, label_r * s, f"{2.0 * radii[i]:.2f}",
            ha=ha, va=va, color="#174ea6", zorder=8, fontproperties=font,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85),
        )


def plot_fitted_circles(out: Path, payload: dict) -> None:
    """画出拟合针齿圆、拟合节圆和设计节圆，并标出节圆误差。"""
    prof = payload["profile"]
    fit = payload["fit"]
    pins = int(fit["pins"])
    rz = float(fit["pin_radius"])
    pitch_radius = float(fit["pitch_radius"])
    offset = np.asarray(fit["offset"], dtype=float)
    ang = np.arange(pins) * (2.0 * np.pi / pins)
    center_r = pitch_radius + offset
    cx = center_r * np.cos(ang)
    cy = center_r * np.sin(ang)

    order = np.argsort(prof["theta"])
    xy = prof["edge"][order]
    ok = np.asarray(prof["valid"][order], dtype=bool)
    px = np.r_[xy[:, 0], xy[0, 0]].astype(float)
    py = np.r_[xy[:, 1], xy[0, 1]].astype(float)
    ok_closed = np.r_[ok, ok[0]]
    px[~ok_closed] = np.nan
    py[~ok_closed] = np.nan

    profile_color = "#4e5156"
    pin_color = "#5b87d6"
    pair = _pitch_pair(payload)
    design_radius = pair[1] if pair else None
    gap_angle = _gap_angle(pins)

    fig_w, fig_h = 1200 / 140, 1420 / 140
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=140, facecolor="white")
    side = 7.15
    left = (fig_w - side) / 2 / fig_w
    bottom = 1.42 / fig_h
    width = side / fig_w
    height = side / fig_h
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_facecolor("white")
    _draw_scan_background(ax, prof["xy"])

    radii = _circle_radii(fit)
    for i in range(pins):
        ax.add_patch(plt.Circle(
            (float(cx[i]), float(cy[i])), float(radii[i]),
            fill=False, ec=pin_color, lw=0.85, zorder=2,
        ))
    _draw_outer_wall(ax, fit, lw=1.25, zorder=4)
    ax.plot(px, py, color=profile_color, lw=1.05, solid_capstyle="round", zorder=3)
    _draw_pitch_circles(ax, pitch_radius, design_radius)
    if design_radius is not None:
        _mark_pitch_gap(ax, pitch_radius, design_radius, gap_angle)
    ax.scatter(cx, cy, s=12, c="#202124", zorder=6, linewidths=0)

    label_r = float(np.max(center_r + radii) + 2.6)
    font_label = _cjk_font(8)
    for i in range(pins):
        if i % 5 != 0:
            continue
        c, s = float(np.cos(ang[i])), float(np.sin(ang[i]))
        if c > 0.25:
            ha = "left"
        elif c < -0.25:
            ha = "right"
        else:
            ha = "center"
        if s > 0.25:
            va = "bottom"
        elif s < -0.25:
            va = "top"
        else:
            va = "center"
        ax.text(
            label_r * c, label_r * s, str(i + 1),
            ha=ha, va=va, color="#202124", zorder=6, fontproperties=font_label,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92),
        )

    lim = label_r + 5.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)", fontproperties=_cjk_font(11), color="#202124")
    ax.set_ylabel("y (mm)", fontproperties=_cjk_font(11), color="#202124")
    ax.tick_params(labelsize=8, colors="#3c4043")
    for spine in ax.spines.values():
        spine.set_color("#dadce0")

    # 图例按列填充，这个顺序排成两行：半径在上，误差在下。
    handles = [
        Line2D([0], [0], color=profile_color, lw=1.3, label="测量内轮廓"),
        Line2D([0], [0], color=_WALL_COLOR, lw=1.6, label="外壁"),
        Line2D([0], [0], color=pin_color, lw=1.4, label="拟合针齿圆（各齿直径）"),
        Line2D([0], [0], color=_FIT_PITCH_COLOR, lw=1.3, ls="--", label=f"拟合节圆 Rz={pitch_radius:.4f} mm"),
    ]
    if design_radius is not None:
        handles.append(Line2D(
            [0], [0], color="#202124", lw=0.9, marker="|", markersize=9,
            label=_pitch_error_text(pitch_radius, design_radius),
        ))
        handles.append(Line2D(
            [0], [0], color=_DESIGN_PITCH_COLOR, lw=1.3,
            label=f"设计节圆 Rz={design_radius:.4f} mm",
        ))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        prop=_cjk_font(10),
        bbox_to_anchor=(0.5, 0.018),
        handlelength=2.6,
        columnspacing=1.4,
    )
    fig.suptitle(
        "拟合针齿圆示意",
        fontproperties=_cjk_font(15),
        color="#202124",
        y=0.972,
    )

    def _place(box: tuple[float, float, float, float]) -> list[float]:
        x, y, w, h = box
        return [left + x * width, bottom + y * height, w * width, h * height]

    pin_box = (0.225, 0.37, 0.26, 0.26)
    axins = fig.add_axes(_place(pin_box))
    axins.set_facecolor("white")
    axins.plot(px, py, color=profile_color, lw=1.35, solid_capstyle="round", zorder=2)
    _draw_outer_wall(axins, fit, lw=1.8, zorder=4)
    axins.add_patch(plt.Circle(
        (float(cx[0]), float(cy[0])), float(radii[0]),
        fill=False, ec=pin_color, lw=1.2, zorder=3,
    ))
    axins.scatter([cx[0]], [cy[0]], s=18, c="#202124", zorder=4, linewidths=0)
    half = float(radii[0]) + 1.75
    axins.set_xlim(float(cx[0]) - half, float(cx[0]) + half)
    axins.set_ylim(float(cy[0]) - half, float(cy[0]) + half)
    axins.set_aspect("equal", adjustable="box")
    axins.set_title("1 号针齿", fontproperties=_cjk_font(9), color="#202124", pad=3)
    axins.tick_params(labelsize=7, colors="#5f6368", length=2.5, pad=1)
    for spine in axins.spines.values():
        spine.set_color("#9aa0a6")
        spine.set_linewidth(0.7)

    if design_radius is not None:
        zoom_box = (0.515, 0.37, 0.26, 0.26)
        axzoom = fig.add_axes(_place(zoom_box))
        _draw_pitch_zoom(axzoom, pitch_radius, design_radius, gap_angle)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fitted_circles.png", dpi=140, facecolor="white")
    plt.close(fig)


def _broken_contour(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按角度连接实测点。步长明显大于正常间距时断开，空档不连线。"""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return np.array([]), np.array([])
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    order = np.argsort(ang)
    p = pts[order]
    a = ang[order]
    if len(p) == 1:
        return p[:, 0].copy(), p[:, 1].copy()
    step = np.diff(a)
    step = step[step > 1e-8]
    if len(step) == 0:
        normal = 2 * np.pi / len(p)
    else:
        ranked = np.sort(step)
        normal = float(np.median(ranked[: max(1, int(len(ranked) * 0.7))]))
    limit = max(normal * 1.8, 1e-6)
    xs = [float(p[0, 0])]
    ys = [float(p[0, 1])]
    for i in range(1, len(p)):
        if a[i] - a[i - 1] > limit:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(float(p[i, 0]))
        ys.append(float(p[i, 1]))
    wrap = (a[0] + 2 * np.pi) - a[-1]
    if wrap <= limit:
        xs.append(float(p[0, 0]))
        ys.append(float(p[0, 1]))
    return np.asarray(xs), np.asarray(ys)


def _disk_radius(pts: np.ndarray) -> float:
    """整图上圆的直径大约 4 像素。"""
    del pts
    return 0.25


def _measured_disks(ax, pts, values, radius, cmap, norm):
    """用数据坐标里的实心圆表示测点，半径单位是毫米。"""
    patches = [Circle((float(x), float(y)), radius) for x, y in np.asarray(pts, dtype=float)]
    coll = PatchCollection(patches, cmap=cmap, norm=norm, zorder=5)
    coll.set_array(np.asarray(values, dtype=float))
    coll.set_edgecolor("#8a8f96")
    coll.set_linewidth(0.15)
    ax.add_collection(coll)
    return coll


def _draw_clearance_zoom(ax, payload: dict) -> None:
    """放大间隙最小的那颗针齿，让圆和外壁之间的空档能看出来。"""
    fit = payload["fit"]
    pts = np.asarray(fit.get("wall_points", np.zeros((0, 2))), dtype=float)
    pin = np.asarray(fit.get("wall_pin", np.zeros(0, dtype=int)))
    pins = int(fit["pins"])
    radii = _circle_radii(fit)
    center_r = np.asarray(fit["pitch_radius"], dtype=float) + np.asarray(fit["offset"], dtype=float)
    ang = np.arange(pins) * (2.0 * np.pi / pins)
    gaps = np.full(pins, np.inf)
    for i in range(pins):
        mine = pts[pin == i] if len(pts) else np.zeros((0, 2))
        if len(mine) == 0 or not np.isfinite(center_r[i]):
            continue
        center = center_r[i] * np.array([np.cos(ang[i]), np.sin(ang[i])])
        gaps[i] = float(np.hypot(mine[:, 0] - center[0], mine[:, 1] - center[1]).min()) - float(radii[i])
    i = int(np.argmin(gaps))
    rz = float(radii[i])
    center = center_r[i] * np.array([np.cos(ang[i]), np.sin(ang[i])])
    cloud = np.asarray(payload["profile"]["xy"], dtype=float)
    near = np.hypot(cloud[:, 0] - center[0], cloud[:, 1] - center[1]) < rz + 1.8
    ax.set_facecolor("white")
    if np.any(near):
        ax.scatter(cloud[near, 0], cloud[near, 1], s=2.0, c="#c5c8ce", linewidths=0, zorder=1, rasterized=True)
    _draw_outer_wall(ax, fit, lw=1.7, zorder=3)
    ax.add_patch(plt.Circle((float(center[0]), float(center[1])), rz, fill=False, ec=_PIN_CIRCLE_COLOR, lw=1.15, zorder=4))
    direction = np.array([np.cos(ang[i]), np.sin(ang[i])])
    focus = center + rz * direction
    span = 1.35
    ax.set_xlim(float(focus[0] - span), float(focus[0] + span))
    ax.set_ylim(float(focus[1] - span), float(focus[1] + span))
    ax.set_aspect("equal", adjustable="box")
    gap = gaps[i]
    title = f"{i + 1} 号  间隙 {gap:.2f} mm" if np.isfinite(gap) else f"{i + 1} 号"
    ax.set_title(title, fontproperties=_cjk_font(8), color="#202124", pad=2)
    ax.tick_params(labelsize=6, colors="#5f6368", length=2, pad=1)
    for spine in ax.spines.values():
        spine.set_color("#9aa0a6")
        spine.set_linewidth(0.6)


def _plot_pin_diameters(out: Path, payload: dict) -> None:
    """每颗针齿一根柱，柱上写该齿的拟合直径。"""
    fit = payload["fit"]
    radii = _circle_radii(fit)
    diam = 2.0 * radii
    pins = len(diam)
    xs = np.arange(1, pins + 1)
    fig, ax = plt.subplots(figsize=(max(11.5, pins * 0.42), 5.4))
    ax.bar(xs, diam, width=0.72, color="#5b87d6", zorder=2)
    top = float(np.max(diam))
    bot = float(np.min(diam))
    span = max(0.35, top - bot)
    ax.set_ylim(bot - 0.55 * span, top + 0.85 * span)
    font = _cjk_font(8)
    for x, d in zip(xs, diam):
        ax.text(
            x, d + 0.03 * span, f"{d:.2f}",
            ha="center", va="bottom", rotation=90, fontsize=8, color="#174ea6", fontproperties=font,
        )
    design = payload.get("design")
    if design and np.isfinite(design.get("pin_radius", np.nan)):
        design_d = 2.0 * float(design["pin_radius"])
        ax.axhline(design_d, color="#d06a00", lw=1.0, ls="--", zorder=1)
        ax.text(
            pins + 0.15, design_d, f"设计 {design_d:.2f}",
            ha="left", va="center", color="#d06a00", fontproperties=_cjk_font(8),
        )
    ax.set_xlim(0.4, pins + 1.6)
    ax.set_xticks(xs)
    ax.set_xlabel("针齿号", fontproperties=_cjk_font(12))
    ax.set_ylabel("拟合直径 (mm)", fontproperties=_cjk_font(12))
    ax.set_title("各针齿拟合直径", fontproperties=_cjk_font(14))
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "pin_diameter.png", dpi=140)
    plt.close(fig)


def plot_report(out: Path, payload: dict) -> None:
    """画齿廓偏差着色图、节圆对比和分齿误差图。"""
    prof = payload["profile"]
    fit = payload["fit"]
    evaluation = payload["eval"]
    err = evaluation["profile_error"]
    valid = np.isfinite(err)
    pts = prof["edge"][valid]
    color = err[valid]
    cmap, norm, ticks, _half = _mm_color_scale(color)
    pins = fit["pins"]
    ref_pitch = evaluation["ref_pitch"]
    ref_pin = evaluation["ref_pin"]

    fig, ax = plt.subplots(figsize=(7.6, 8.3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    _draw_scan_background(ax, prof["xy"])
    curve = _design_curve(pins, ref_pitch, ref_pin)
    ax.plot(curve[:, 0], curve[:, 1], color="#9aa0a6", lw=1.0, zorder=2)
    measured_pitch = float(fit["pitch_radius"])
    design_pitch = float(payload["design"]["pitch_radius"]) if payload.get("design") else None
    _draw_pitch_circles(ax, measured_pitch, design_pitch, fitted_lw=1.05)
    if design_pitch is not None:
        _mark_pitch_gap(ax, measured_pitch, design_pitch, _gap_angle(int(pins)))
    outline_x, outline_y = _broken_contour(pts)
    ax.plot(outline_x, outline_y, color="#4a4d51", lw=0.7, solid_capstyle="round", zorder=4)
    point_radius = _disk_radius(pts)
    sc = _measured_disks(ax, pts, color, point_radius, cmap, norm)
    _draw_pin_circles(ax, fit, lw=0.85, zorder=6.4)
    _draw_outer_wall(ax, fit, lw=1.15, zorder=6.6)
    centers = (fit["pitch_radius"] + fit["offset"])[:, None] * np.column_stack([
        np.cos(np.arange(pins) * 2 * np.pi / pins),
        np.sin(np.arange(pins) * 2 * np.pi / pins),
    ])
    ax.scatter(centers[:, 0], centers[:, 1], c="#202124", s=10, zorder=8, linewidths=0)
    _draw_diameter_labels(ax, fit)
    cb = _banded_colorbar(fig, sc, ax, ticks, fraction=0.046, pad=0.04)
    cb.set_label("齿廓偏差 (mm)", fontproperties=_cjk_font(11))
    cb.ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontproperties=_cjk_font(11))
    ax.set_ylabel("y (mm)", fontproperties=_cjk_font(11))
    ax.set_title(
        f"针齿齿廓   偏差 RMS {evaluation['profile_stats']['rms']:.4f} mm",
        fontproperties=_cjk_font(13),
    )
    overlay_handles = [
        Line2D(
            [0], [0], marker="o", ls="none", markersize=5,
            markerfacecolor="#c5c9ce", markeredgecolor="#8a8f96", markeredgewidth=0.5,
            label=f"测点半径 {point_radius:.2f} mm",
        ),
        Line2D([0], [0], color="#9aa0a6", lw=1.2, label="理论齿廓"),
        Line2D([0], [0], color=_WALL_COLOR, lw=1.4, label="外壁"),
        Line2D([0], [0], color=_PIN_CIRCLE_COLOR, lw=1.2, label="拟合针齿圆"),
        Line2D([0], [0], color="#202124", marker="o", ls="none", markersize=4, label="针齿中心"),
        Line2D([0], [0], color=_FIT_PITCH_COLOR, lw=1.2, ls="--", label=f"拟合节圆 Rz={measured_pitch:.4f} mm"),
    ]
    if design_pitch is not None:
        overlay_handles.append(Line2D(
            [0], [0], color="#202124", lw=0.9, marker="|", markersize=8,
            label=_pitch_error_text(measured_pitch, design_pitch),
        ))
        overlay_handles.append(Line2D(
            [0], [0], color=_DESIGN_PITCH_COLOR, lw=1.2,
            label=f"设计节圆 Rz={design_pitch:.4f} mm",
        ))
    if design_pitch is not None:
        zoom = ax.inset_axes([0.36, 0.38, 0.28, 0.28])
        _draw_pitch_zoom(zoom, measured_pitch, design_pitch, _gap_angle(int(pins)))
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    fig.legend(
        handles=overlay_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        prop=_cjk_font(8.5),
        bbox_to_anchor=(0.45, 0.012),
        handlelength=2.2,
        columnspacing=1.2,
    )
    fig.savefig(out / "overlay.png", dpi=140, facecolor="white")
    plt.close(fig)

    ang = np.degrees(prof["theta"][valid])
    order = np.argsort(ang)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axhline(0, color="#888", lw=0.8)
    xy = np.column_stack([ang[order], color[order]])
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    line = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.3)
    line.set_array(0.5 * (color[order][:-1] + color[order][1:]))
    ax.add_collection(line)
    ax.set_xlim(float(ang[order][0]), float(ang[order][-1]))
    span = float(np.max(np.abs(color))) if len(color) else _half
    ax.set_ylim(-(span + 0.05), span + 0.05)
    ax.set_xlabel("角度 (°)")
    ax.set_ylabel("齿廓偏差 (mm)")
    ax.set_title("齿廓偏差沿圆周的分布")
    cb_line = _banded_colorbar(fig, line, ax, ticks, fraction=0.03, pad=0.02)
    cb_line.set_label("齿廓偏差 (mm)", fontproperties=_cjk_font(11))
    cb_line.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "error_vs_angle.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(color, bins=60, color="#1a73e8", edgecolor="white")
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("齿廓偏差 (mm)")
    ax.set_ylabel("采样点数")
    ax.set_title("齿廓偏差分布")
    fig.tight_layout()
    fig.savefig(out / "error_histogram.png", dpi=140)
    plt.close(fig)

    rows = evaluation["per_pin"]
    fig, ax = plt.subplots(figsize=(max(8, pins * 0.28), 4.4))
    xs = np.arange(1, pins + 1)
    pitch_err = [row["pitch_error"] for row in rows]
    means = [row["profile"]["mean"] for row in rows]
    ax.bar(xs - 0.18, pitch_err, width=0.36, color="#188038", label="节圆径向偏差")
    ax.bar(xs + 0.18, means, width=0.36, color="#1a73e8", label="齿廓平均偏差")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("针齿号")
    ax.set_ylabel("偏差 (mm)")
    ax.set_title("各针齿的节圆偏差和齿廓偏差")
    ax.set_xticks(xs)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_per_pin.png", dpi=140)
    plt.close(fig)
    _plot_pin_diameters(out, payload)
    plot_fitted_circles(out, payload)


def write_profile_ply(path: Path, pts: np.ndarray, err: np.ndarray) -> None:
    """把带偏差颜色的齿廓点写成 ASCII PLY。"""
    lim = _limit(err)
    t = np.clip(err / lim, -1, 1)
    rgb = np.zeros((len(pts), 3))
    rgb[:, 0] = np.clip(1 + t, 0, 1)
    rgb[:, 1] = 1 - np.abs(t)
    rgb[:, 2] = np.clip(1 - t, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, rgb):
            f.write(f"{p[0]:.5f} {p[1]:.5f} 0 {c[0]} {c[1]} {c[2]}\n")


# ---------------------------------------------------------------------------
# 自检与主流程
# ---------------------------------------------------------------------------

def _synthetic_cloud(pitch_radius: float, pin_radius: float, pins: int, offset: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    # 齿侧在一个角度扇区里半径变化大。点取得密一些，提取出的边缘才是齿侧而不是扇区内侧。
    """按给定节圆、针齿半径和各齿偏移生成仿真点云，供自检使用。"""
    theta = np.linspace(-np.pi, np.pi, pins * 720, endpoint=False)
    radius = np.full(len(theta), pitch_radius)
    direction = np.column_stack([np.cos(theta), np.sin(theta)])
    pitch = 2 * np.pi / pins
    for i in range(pins):
        center = (pitch_radius + offset[i]) * np.array([np.cos(i * pitch), np.sin(i * pitch)])
        proj = direction @ center
        disc = proj ** 2 - np.dot(center, center) + pin_radius ** 2
        hit = disc > 0
        far = proj.copy()
        far[hit] = proj[hit] + np.sqrt(disc[hit])
        radius = np.maximum(radius, far)
    radius = radius + rng.normal(0.0, noise, len(radius))
    clouds = [np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])]
    for extra in (0.25, 0.7, 1.3, 2.0):
        rr = radius + extra
        clouds.append(np.column_stack([rr * np.cos(theta), rr * np.sin(theta)]))
    hole_ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    local = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    for ang in hole_ang:
        center = (pitch_radius * 0.62) * np.array([np.cos(ang), np.sin(ang)])
        clouds.append(center + 2.2 * np.column_stack([np.cos(local), np.sin(local)]))
    xy = np.vstack(clouds)
    xy = rotate(xy, np.deg2rad(17.0)) + np.array([0.4, -0.25])
    return xy


def self_test() -> None:
    """用已知误差的仿真齿廓检查拟合能否找回节圆误差、针齿半径误差和各齿跳动。"""
    rng = np.random.default_rng(7)
    pins = 31
    design = design_from_parameters(pins, pitch_diameter=85.0, pin_diameter=5.0)
    # 一次谐波相当于节圆圆心没放正，程序会把它归到坐标系里。
    # 这里只放二次、三次谐波，检查各针齿相对节圆的跳动还能不能找回来。
    offset = 0.02 * np.sin(np.arange(pins) * 4 * np.pi / pins) + 0.01 * np.cos(np.arange(pins) * 6 * np.pi / pins)
    offset = offset - offset.mean()
    true_pitch = design["pitch_radius"] + 0.05
    true_pin = design["pin_radius"] - 0.02
    cloud = _synthetic_cloud(true_pitch, true_pin, pins, offset, noise=0.002, rng=rng)
    result = analyze_xy(cloud, design, n_bins=1440, pins=pins)
    fit = result["fit"]
    got = fit["offset"]
    pitch_err = fit["pitch_radius"] - design["pitch_radius"]
    pin_err = fit["pin_radius"] - design["pin_radius"]
    # 齿号可能整体错开一颗，取圆周相关最大的对齐
    shift = int(np.argmax([np.dot(got, np.roll(offset, k)) for k in range(pins)]))
    got = np.roll(got, -shift)
    dr_rms = float(np.sqrt(np.mean((got - offset) ** 2)))
    print(
        f"自检 节圆半径误差 {pitch_err:+.4f} mm（期望 +0.050），"
        f"针齿半径误差 {pin_err:+.4f} mm（期望 -0.020），"
        f"各齿径向偏差 RMS {dr_rms:.4f} mm"
    )
    if abs(pitch_err - 0.05) > 0.02:
        raise SystemExit(f"自检失败：节圆半径误差 {pitch_err:+.4f}，期望约 +0.050")
    if abs(pin_err + 0.02) > 0.02:
        raise SystemExit(f"自检失败：针齿半径误差 {pin_err:+.4f}，期望约 -0.020")
    if dr_rms > 0.012:
        raise SystemExit(f"自检失败：各针齿径向偏差 RMS {dr_rms:.4f}，期望小于 0.012")
    print("自检通过")


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数：点云、模型或设计参数、针齿数和输出目录。"""
    parser = argparse.ArgumentParser(description="计算摆线减速机针齿齿廓误差和节圆半径误差")
    parser.add_argument("--scan", type=Path, default=Path("2.ply"), help="测量点云 PLY，默认 2.ply")
    parser.add_argument("--model", type=Path, default=None, help="针齿壳名义模型，STL / OBJ / PLY")
    parser.add_argument("--pins", type=int, default=None, help="针齿数。用设计参数时必填，等于摆线轮齿数 + 1")
    parser.add_argument("--pitch-diameter", type=float, default=None, help="节圆直径（针齿中心圆直径），毫米")
    parser.add_argument("--pin-diameter", type=float, default=None, help="针齿直径，毫米")
    parser.add_argument("--bins", type=int, default=1440, help="轮廓角度采样数，默认 1440（0.25°）")
    parser.add_argument("--tolerance", type=float, default=None, help="齿廓公差，毫米。给出后统计超差比例")
    parser.add_argument("--out", type=Path, default=Path("pin_report"), help="结果目录")
    parser.add_argument("--self-test", action="store_true", help="用带已知误差的仿真齿廓检查程序")
    return parser


def main(argv: list[str] | None = None) -> None:
    """读参数后跑自检，或计算一次针齿齿廓误差并写出报告。"""
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return
    if not args.scan.is_file():
        raise SystemExit(f"找不到测量点云: {args.scan}")
    given = [args.pins, args.pitch_diameter, args.pin_diameter]
    has_params = any(v is not None for v in given)
    if args.model and has_params:
        raise SystemExit("请只指定一种设计值：--model，或者 --pins --pitch-diameter --pin-diameter。")
    if has_params and any(v is None for v in given):
        raise SystemExit("设计参数需要同时给出 --pins --pitch-diameter --pin-diameter。")

    points, _ = read_ply(args.scan)
    xy = project_scan(points)
    design = None
    if args.model:
        if not args.model.is_file():
            raise SystemExit(f"找不到三维模型: {args.model}")
        design, _ = design_from_mesh(load_model(args.model), rough_radius(points), args.bins)
    elif has_params:
        design = design_from_parameters(args.pins, args.pitch_diameter, args.pin_diameter)

    result = analyze_xy(xy, design, args.bins, pins=None if design is None else design["pins"])
    result["scan"] = str(args.scan)
    result["n_points"] = int(len(points))
    result["tolerance"] = args.tolerance
    write_report(args.out, result)
    plot_report(args.out, result)
    err = result["eval"]["profile_error"]
    valid = np.isfinite(err)
    write_profile_ply(args.out / "profile_deviation.ply", result["profile"]["edge"][valid], err[valid])


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
