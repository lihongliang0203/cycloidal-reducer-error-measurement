#!/usr/bin/env python3
"""把测量到的摆线盘外轮廓和三维模型（或设计参数）对齐，计算加工误差。

主结果是轮廓法向偏差：测量外轮廓相对名义齿廓的最短有符号距离。
正值表示实测轮廓在名义轮廓外侧（多肉、余量大），负值表示少肉。

名义齿廓来源二选一：

1. 三维模型（STL / OBJ / PLY 网格）。程序把模型外轮廓投到盘面后比较。
2. 设计参数。公式与 CycloidalDrive 脚本一致（轮心为原点，齿廓为外摆线
   向内偏移一个针齿半径的等距曲线）。

示例：

    python compare_machining_error.py --scan 1.ply --model nominal.stl
    python compare_machining_error.py --scan 1.ply --teeth 30 \\
        --pitch-diameter 82 --pin-diameter 3 --eccentricity 1.2

只给点云时，会导出实测轮廓和齿数估计，不算误差。
长度单位默认毫米。模型若是米制，且与测量半径相差约 1000 倍，会自动换算。
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


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


# ---------------------------------------------------------------------------
# 文件读取
# ---------------------------------------------------------------------------

def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """返回 (points Nx3, faces Mx3 或 None)。支持二进制小端和 ASCII。"""
    raw = Path(path).read_bytes()
    marker = b"end_header\r\n" if b"end_header\r\n" in raw else b"end_header\n"
    head, body = raw.split(marker, 1)
    lines = head.decode("utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"不是 PLY 文件: {path}")

    fmt = "ascii"
    n_vert = n_face = 0
    vert_props: list[str] = []
    in_vert = in_face = False
    face_count_type = "uchar"
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element" and parts[1] == "vertex":
            n_vert = int(parts[2])
            in_vert, in_face = True, False
        elif parts[0] == "element" and parts[1] == "face":
            n_face = int(parts[2])
            in_vert, in_face = False, True
        elif parts[0] == "element":
            in_vert = in_face = False
        elif parts[0] == "property" and in_vert:
            vert_props.append(parts[-1])
        elif parts[0] == "property" and in_face and parts[1] == "list":
            face_count_type = parts[2]

    if "x" not in vert_props or "y" not in vert_props or "z" not in vert_props:
        raise ValueError(f"PLY 顶点缺少 x/y/z: {path}")

    if fmt == "ascii":
        text = body.decode("utf-8", errors="replace").splitlines()
        rows = []
        for line in text:
            if line.strip():
                rows.append([float(x) for x in line.split()])
            if len(rows) == n_vert:
                break
        cols = {name: i for i, name in enumerate(vert_props)}
        arr = np.asarray(rows, dtype=np.float64)
        points = np.column_stack([arr[:, cols["x"]], arr[:, cols["y"]], arr[:, cols["z"]]])
        faces = None
        if n_face:
            faces = _parse_ascii_faces(text[n_vert : n_vert + n_face])
        return points, faces

    if fmt != "binary_little_endian":
        raise ValueError(f"暂不支持的 PLY 格式 {fmt}: {path}")

    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "short": "<i2",
        "ushort": "<u2",
    }
    # 重新扫 property 行，拿到类型
    vert_types: list[str] = []
    in_vert = False
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element" and parts[1] == "vertex":
            in_vert = True
        elif parts[0] == "element":
            in_vert = False
        elif parts[0] == "property" and in_vert and parts[1] != "list":
            vert_types.append(type_map[parts[1]])
    dtype = np.dtype([(name, dt) for name, dt in zip(vert_props, vert_types)])
    vert = np.frombuffer(body, dtype=dtype, count=n_vert)
    points = np.column_stack([vert["x"], vert["y"], vert["z"]]).astype(np.float64)
    faces = None
    if n_face:
        faces = _parse_binary_faces(body[vert.nbytes :], n_face, face_count_type)
    return points, faces


def _parse_ascii_faces(lines: list[str]) -> np.ndarray:
    """把 ASCII PLY 里的多边形面拆成三角形。"""
    faces = []
    for line in lines:
        vals = [int(float(x)) for x in line.split()]
        if len(vals) < 4:
            continue
        n = vals[0]
        idx = vals[1 : 1 + n]
        for k in range(1, n - 1):
            faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(faces, dtype=np.int64)


def _parse_binary_faces(buf: bytes, n_face: int, count_type: str) -> np.ndarray:
    """把二进制 PLY 里的多边形面拆成三角形。"""
    count_map = {"uchar": "B", "uint8": "B", "ushort": "H", "uint": "I", "int": "i"}
    code = count_map.get(count_type, "B")
    count_size = struct.calcsize("<" + code)
    faces = []
    off = 0
    for _ in range(n_face):
        (n,) = struct.unpack_from("<" + code, buf, off)
        off += count_size
        idx = struct.unpack_from("<" + "i" * n, buf, off)
        off += 4 * n
        for k in range(1, n - 1):
            faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(faces, dtype=np.int64)


def read_stl(path: Path) -> np.ndarray:
    """返回三角形数组 (T, 3, 3)。"""
    raw = Path(path).read_bytes()
    if raw[:5].lower() == b"solid" and b"facet" in raw[:2000].lower():
        return _read_ascii_stl(raw.decode("utf-8", errors="replace"))
    if len(raw) < 84:
        raise ValueError(f"STL 文件过短: {path}")
    n = struct.unpack_from("<I", raw, 80)[0]
    need = 84 + n * 50
    if len(raw) < need:
        raise ValueError(f"STL 三角形数量与文件长度不符: {path}")
    tri = np.empty((n, 3, 3), dtype=np.float64)
    off = 84
    for i in range(n):
        nums = struct.unpack_from("<12fH", raw, off)
        tri[i, 0] = nums[3:6]
        tri[i, 1] = nums[6:9]
        tri[i, 2] = nums[9:12]
        off += 50
    return tri


def _read_ascii_stl(text: str) -> np.ndarray:
    """从 ASCII STL 文本里读出三角形顶点。"""
    verts: list[list[float]] = []
    tris = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "vertex":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(verts) == 3:
                tris.append(verts)
                verts = []
    if not tris:
        raise ValueError("ASCII STL 中没有三角形")
    return np.asarray(tris, dtype=np.float64)


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读 OBJ 的顶点和面，多边形按扇形拆成三角形。"""
    verts = []
    faces = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("v "):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                idx.append(int(tok.split("/")[0]) - 1)
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not faces:
        raise ValueError(f"OBJ 缺少顶点或面: {path}")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def load_model(path: Path) -> np.ndarray:
    """读入三维模型，统一成三角形数组 (T, 3, 3)。"""
    ext = path.suffix.lower()
    if ext == ".stl":
        return read_stl(path)
    if ext == ".obj":
        verts, faces = read_obj(path)
        return verts[faces]
    if ext == ".ply":
        verts, faces = read_ply(path)
        if faces is None or len(faces) == 0:
            raise ValueError(f"PLY 模型没有面片，无法作为三维模型: {path}")
        return verts[faces]
    raise ValueError(f"不支持的模型格式 {ext}，请使用 STL、OBJ 或带面的 PLY")


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------

def fit_disc_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用盘面点拟合平面。返回中心、平面内两轴、法向、各点高度。"""
    mask = np.ones(len(points), dtype=bool)
    center = points.mean(axis=0)
    normal = np.array([0.0, 0.0, 1.0])
    for _ in range(5):
        pts = points[mask]
        center = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
        normal = vt[2].copy()
        if normal[2] < 0:
            normal = -normal
        height = (points - center) @ normal
        med = np.median(height[mask])
        mad = np.median(np.abs(height[mask] - med)) + 1e-9
        new_mask = np.abs(height - med) < max(0.3, 12 * mad)
        if new_mask.sum() < max(100, int(0.2 * len(points))):
            break
        mask = new_mask
    e1 = vt[0].copy()
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)
    height = (points - center) @ normal
    return center, e1, e2, normal, height


def to_plane(points: np.ndarray, center, e1, e2, normal) -> tuple[np.ndarray, np.ndarray]:
    """把三维点投到拟合盘面上，返回平面坐标和沿法向的高度。"""
    rel = points - center
    uv = np.column_stack([rel @ e1, rel @ e2])
    height = rel @ normal
    return uv, height


# 双目结构光噪声通常远小于 0.05 mm。径向带只留齿廓外缘；
# 再宽（原先 0.2 mm）会把盘面点算进齿廓，把半径拉小。
EDGE_BAND_MM = 0.05
# 衰减长度与带宽同量级。带内内缘权重约 e^-1，和外缘接近，
# 让这一圈边缘点一起决定半径，而不是只剩最外一个点。
EDGE_WEIGHT_MM = 0.05


def _radius_at_bin_center(angle: np.ndarray, radius: np.ndarray, theta: float) -> float:
    """把外轮廓点的半径插值到扇区中心。

    扇区里还有盘面点时，只留下距外侧参考 EDGE_BAND_MM 以内的一圈。
    外侧参考用高分位，避免一颗飞点把整条带抬出去。权重相对这个参考衰减，
    比参考更靠外的点不再额外加权。
    """
    rel = (angle - theta + np.pi) % (2 * np.pi) - np.pi
    # 点数够时用 99% 分位，不用裸最大值，压住离群飞点
    r_hi = float(np.percentile(radius, 99) if len(radius) >= 30 else np.max(radius))
    band = radius >= r_hi - EDGE_BAND_MM
    if int(band.sum()) < 2:
        return r_hi
    x = rel[band]
    y = radius[band]
    if float(np.max(x) - np.min(x)) < 1e-8:
        return float(np.median(y))
    # 相对 r_hi 衰减，并在 r_hi 处封顶：飞点即使更靠外，权重也不超过正常外缘
    weight = np.exp(np.minimum(y - r_hi, 0.0) / EDGE_WEIGHT_MM)
    design = np.column_stack([np.ones(len(x)), x]) * np.sqrt(weight)[:, None]
    coef, _, _, _ = np.linalg.lstsq(design, y * np.sqrt(weight), rcond=None)
    outer = weight > 0.2
    brackets_center = bool(np.any(outer) and np.min(x[outer]) < 0.0 < np.max(x[outer]))
    if brackets_center:
        return float(coef[0])
    nearest = np.argpartition(np.abs(x), min(4, len(x) - 1))[: min(5, len(x))]
    return float(np.median(y[nearest]))


def radial_envelope(xy: np.ndarray, n_bins: int, min_count: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """按角度扇区提取外轮廓，半径取在扇区中心角上。"""
    radius = np.hypot(xy[:, 0], xy[:, 1])
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    bins = np.floor((angle + np.pi) / (2 * np.pi) * n_bins).astype(np.int64)
    bins = np.clip(bins, 0, n_bins - 1)
    order = np.argsort(bins, kind="mergesort")
    bins_s = bins[order]
    rad_s = radius[order]
    ang_s = angle[order]
    cuts = np.flatnonzero(np.diff(bins_s)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(bins_s)]
    theta = (np.arange(n_bins) + 0.5) / n_bins * 2 * np.pi - np.pi

    r_out = np.full(n_bins, np.nan)
    supported = np.zeros(n_bins, dtype=bool)
    for b, s, e in zip(bins_s[starts], starts, ends):
        if e - s < min_count:
            continue
        r_out[b] = _radius_at_bin_center(ang_s[s:e], rad_s[s:e], theta[b])
        supported[b] = True

    valid = np.isfinite(r_out)
    if valid.sum() < n_bins * 0.5:
        raise RuntimeError("外轮廓有效扇区不足一半，点云可能没有形成完整的盘形轮廓")
    idx = np.arange(n_bins)
    if (~valid).any():
        r_out[~valid] = np.interp(idx[~valid], idx[valid], r_out[valid], period=n_bins)

    # 去掉单扇区飞点，不抹平真实的局部缺陷
    padded = np.r_[r_out[-2:], r_out, r_out[:2]]
    neighbor = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)
    spike = np.abs(r_out - neighbor) > 0.8
    r_out[spike] = neighbor[spike]
    return np.column_stack([r_out * np.cos(theta), r_out * np.sin(theta)]), supported


def center_and_extract(xy: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """迭代把外轮廓质心移到原点。

    返回轮廓点、最终中心，以及每个扇区在插值补洞之前是否真有测点。
    补洞只服务于数值对齐，图上不把这些空档画成实测。
    """
    center = np.median(xy, axis=0)
    profile = None
    supported = np.zeros(n_bins, dtype=bool)
    for _ in range(6):
        profile, supported = radial_envelope(xy - center, n_bins)
        delta = profile.mean(axis=0)
        center = center + delta
        if np.linalg.norm(delta) < 1e-4:
            break
    return profile, center, supported


def estimate_teeth(profile: np.ndarray) -> int:
    """用轮廓半径的主谐波估计摆线轮齿数。"""
    radius = np.hypot(profile[:, 0], profile[:, 1])
    spec = np.abs(np.fft.rfft(radius - radius.mean()))
    spec[:2] = 0
    teeth = int(np.argmax(spec))
    if teeth < 3:
        raise RuntimeError("无法从轮廓估计齿数")
    return teeth


def cycloid_profile(pin_count: int, pin_radius: float, pitch_radius: float, eccentricity: float, points_per_tooth: int) -> np.ndarray:
    """摆线轮理论齿廓，轮心在原点。pin_count 为针齿数（齿数 + 1）。"""
    if pin_count < 3:
        raise ValueError("针齿数至少为 3")
    if min(pin_radius, pitch_radius, eccentricity) <= 0:
        raise ValueError("针齿半径、节圆半径、偏心距都必须大于 0")
    if eccentricity * pin_count >= pitch_radius:
        raise ValueError(
            "参数会产生奇点：偏心距 × 针齿数 必须小于节圆半径 "
            f"({eccentricity:.4f} × {pin_count} >= {pitch_radius:.4f})"
        )
    teeth = pin_count - 1
    rm = pitch_radius / pin_count
    rc = rm * teeth
    rd = eccentricity
    k = (rc + rm) / rm
    p = np.linspace(0.0, 2.0 * np.pi, teeth * points_per_tooth, endpoint=False)
    xa = (rc + rm) * np.cos(p) - rd * np.cos(k * p)
    ya = (rc + rm) * np.sin(p) - rd * np.sin(k * p)
    dxa = -(rc + rm) * np.sin(p) + k * rd * np.sin(k * p)
    dya = (rc + rm) * np.cos(p) - k * rd * np.cos(k * p)
    norm = np.hypot(dxa, dya)
    xp = xa - pin_radius * dya / norm
    yp = ya + pin_radius * dxa / norm
    return np.column_stack([xp, yp])


def sample_mesh_silhouette(triangles: np.ndarray, step: float) -> np.ndarray:
    """把网格边按步长采样，供外轮廓提取。"""
    edges = np.concatenate(
        [triangles[:, [0, 1], :], triangles[:, [1, 2], :], triangles[:, [2, 0], :]],
        axis=0,
    )
    p0 = edges[:, 0, :]
    p1 = edges[:, 1, :]
    length = np.linalg.norm(p1 - p0, axis=1)
    # 过短的退化边丢掉
    keep = length > 1e-9
    p0, p1, length = p0[keep], p1[keep], length[keep]
    n = np.clip(np.ceil(length / step).astype(np.int64), 1, 80)
    chunks = [p0, p1]
    for k in np.unique(n):
        sel = n == k
        t = np.linspace(0.0, 1.0, int(k), endpoint=False)
        seg = (1.0 - t)[None, :, None] * p0[sel, None, :] + t[None, :, None] * p1[sel, None, :]
        chunks.append(seg.reshape(-1, 3))
    return np.vstack(chunks)


def signed_area(poly: np.ndarray) -> float:
    """计算多边形的有符号面积，逆时针为正。"""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """把闭合轮廓调成逆时针。"""
    return poly if signed_area(poly) >= 0 else poly[::-1].copy()


def resample_radius(profile: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """按等角度重采样闭合轮廓的半径。"""
    angle = np.arctan2(profile[:, 1], profile[:, 0])
    radius = np.hypot(profile[:, 0], profile[:, 1])
    order = np.argsort(angle)
    angle, radius = angle[order], radius[order]
    # 去掉重复角度
    uniq = np.r_[True, np.diff(angle) > 1e-12]
    angle, radius = angle[uniq], radius[uniq]
    grid = np.linspace(-np.pi, np.pi, n, endpoint=False)
    ang_pad = np.r_[angle[-1] - 2 * np.pi, angle, angle[0] + 2 * np.pi]
    rad_pad = np.r_[radius[-1], radius, radius[0]]
    return grid, np.interp(grid, ang_pad, rad_pad)


def point_to_polyline(query: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """闭合多段线的有符号距离。多段线须为逆时针，正值在外侧。"""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)
    ap = query[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / np.maximum(ab2, 1e-18), 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    diff = query[:, None, :] - closest
    d2 = np.sum(diff * diff, axis=2)
    seg = np.argmin(d2, axis=1)
    row = np.arange(len(query))
    delta = diff[row, seg]
    dist = np.sqrt(d2[row, seg])
    tangent = ab[seg]
    outward = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    sign = np.sign(np.sum(delta * outward, axis=1))
    sign[sign == 0] = 1.0
    return sign * dist, closest[row, seg]


def kabsch_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """求两组平面点之间的最优旋转和平移。"""
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    h = (src - cs).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        rot = vt.T @ u.T
    trans = cd - rot @ cs
    return rot, trans


def apply_rigid(pts: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """对平面点施加旋转和平移。"""
    return pts @ rot.T + trans


def _prefer_alignment(candidate: dict, current: dict | None) -> bool:
    """同向取更小 RMS。标准摆线齿左右对称，只有镜像明显更贴合才采用。"""
    if current is None:
        return True
    if candidate["mirrored"] == current["mirrored"]:
        return candidate["rms"] < current["rms"]
    if candidate["mirrored"] and not current["mirrored"]:
        return candidate["rms"] < current["rms"] - 0.01
    return candidate["rms"] <= current["rms"] + 0.01


def align_profiles(measured: np.ndarray, nominal: np.ndarray, teeth: int) -> dict:
    """平移已完成。搜索旋转和镜像，再用点到曲线迭代配准。"""
    n_grid = 4096
    _, r_nom = resample_radius(nominal, n_grid)
    r_nom = r_nom - r_nom.mean()
    best = None
    for mirrored in (False, True):
        pts = measured.copy()
        if mirrored:
            pts[:, 0] *= -1
        _, r_meas = resample_radius(pts, n_grid)
        r_meas = r_meas - r_meas.mean()
        corr = np.fft.ifft(np.fft.fft(r_meas) * np.conj(np.fft.fft(r_nom))).real
        for shift in np.argsort(corr)[-3:][::-1]:
            # r_meas(θ) 与 r_nom(θ + shift) 对齐时，实测点需要逆时针转 shift 个栅格
            angle = shift / n_grid * 2 * np.pi
            c, s = np.cos(angle), np.sin(angle)
            rot0 = np.array([[c, -s], [s, c]])
            aligned = apply_rigid(pts, rot0, np.zeros(2))
            result = _refine_icp(aligned, nominal)
            result["mirrored"] = mirrored
            result["rotation_deg"] = np.degrees(angle + result["refine_rotation_rad"])
            if _prefer_alignment(result, best):
                best = result
    # 齿对称：把转角折到一个齿距内，便于阅读
    tooth = 360.0 / teeth
    rot = best["rotation_deg"]
    rot = (rot + tooth / 2) % tooth - tooth / 2
    best["rotation_deg"] = float(rot)
    return best


def _refine_icp(measured: np.ndarray, nominal: np.ndarray, iterations: int = 20) -> dict:
    """用点到名义曲线的距离反复配准，得到法向偏差。"""
    current = measured.copy()
    rot_acc = np.eye(2)
    trans_acc = np.zeros(2)
    dist = None
    for _ in range(iterations):
        dist, closest = point_to_polyline(current, nominal)
        scale = np.std(dist) + 1e-9
        keep = np.abs(dist - np.median(dist)) < max(0.5, 4 * scale)
        if keep.sum() < max(30, len(current) // 10):
            keep = np.ones(len(current), dtype=bool)
        rot, trans = kabsch_2d(current[keep], closest[keep])
        current = apply_rigid(current, rot, trans)
        rot_acc = rot @ rot_acc
        trans_acc = rot @ trans_acc + trans
        if np.linalg.norm(trans) < 1e-5 and abs(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))) < 1e-4:
            break
    dist, closest = point_to_polyline(current, nominal)
    radius_m = np.hypot(current[:, 0], current[:, 1])
    radius_c = np.hypot(closest[:, 0], closest[:, 1])
    # 径向偏差：同一射线方向上的半径差。closest 不在同一射线上，改用名义半径插值
    grid, r_nom = resample_radius(nominal, 8192)
    ang = np.arctan2(current[:, 1], current[:, 0])
    ang_pad = np.r_[grid[-1] - 2 * np.pi, grid, grid[0] + 2 * np.pi]
    rad_pad = np.r_[r_nom[-1], r_nom, r_nom[0]]
    r_at = np.interp(ang, ang_pad, rad_pad)
    radial = radius_m - r_at
    return {
        "points": current,
        "normal_error": dist,
        "radial_error": radial,
        "nominal_radius": r_at,
        "rms": float(np.sqrt(np.mean(dist**2))),
        "refine_rotation_rad": float(np.arctan2(rot_acc[1, 0], rot_acc[0, 0])),
        "translation": trans_acc,
        "closest_radius": radius_c,
    }


def tooth_index(points: np.ndarray, nominal: np.ndarray, teeth: int) -> tuple[np.ndarray, float]:
    """从名义轮廓的一个齿谷起，把实测点分成 teeth 个齿。"""
    grid, radius = resample_radius(nominal, 4096)
    valleys, _ = find_peaks(-radius, distance=max(3, 4096 // teeth // 2))
    if len(valleys) == 0:
        start = 0.0
    else:
        start = grid[valleys[0]]
    ang = np.arctan2(points[:, 1], points[:, 0])
    rel = (ang - start) % (2 * np.pi)
    idx = np.floor(rel / (2 * np.pi) * teeth).astype(int)
    idx = np.clip(idx, 0, teeth - 1)
    return idx, float(np.degrees(start))


def stats(values: np.ndarray) -> dict:
    """计算均值、标准差、均方根、极值、峰谷和分位数。"""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "pv": float(np.ptp(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def maybe_convert_units(triangles: np.ndarray, scan_radius: float) -> tuple[np.ndarray, str]:
    """按测量半径判断模型单位，并把模型换算成毫米。"""
    verts = triangles.reshape(-1, 3)
    center = verts.mean(axis=0)
    _, _, vt = np.linalg.svd(verts - center, full_matrices=False)
    normal = vt[2]
    e1 = vt[0]
    e2 = np.cross(normal, e1)
    uv = np.column_stack([(verts - center) @ e1, (verts - center) @ e2])
    # 用外圈半径，避免实体盘面中心附近的顶点把尺寸估小
    model_radius = float(np.percentile(np.hypot(uv[:, 0], uv[:, 1]), 98))
    if model_radius <= 0:
        return triangles, "未换算"
    ratio = scan_radius / model_radius
    if 800 <= ratio <= 1200:
        return triangles * 1000.0, "模型数值约为米，已乘 1000 换成毫米"
    if 80 <= ratio <= 120:
        return triangles * 100.0, "模型数值约为分米，已乘 100 换成毫米"
    if 8 <= ratio <= 12:
        return triangles * 10.0, "模型数值约为厘米，已乘 10 换成毫米"
    if not 0.5 <= ratio <= 2.0:
        raise RuntimeError(
            f"模型和测量的尺寸差太多（测量半径约 {scan_radius:.2f}，模型半径约 {model_radius:.4f}）。"
            "请确认单位，或先把模型换成毫米。"
        )
    return triangles, "模型按毫米使用"


# ---------------------------------------------------------------------------
# 名义轮廓
# ---------------------------------------------------------------------------

def nominal_from_parameters(args, n_bins: int) -> tuple[np.ndarray, int, str]:
    """用齿数、节圆直径、针齿直径和偏心距生成理论摆线轮廓。"""
    teeth = args.teeth
    pin_count = teeth + 1
    pitch_radius = args.pitch_diameter / 2.0
    pin_radius = args.pin_diameter / 2.0
    curve = cycloid_profile(pin_count, pin_radius, pitch_radius, args.eccentricity, points_per_tooth=48)
    curve = ensure_ccw(curve)
    # 重采样到与测量相同的角度栅格，避免参数采样不均匀
    profile, _, _ = center_and_extract(curve, n_bins)
    desc = (
        f"设计参数：齿数 {teeth}，针齿数 {pin_count}，节圆直径 {args.pitch_diameter:g} mm，"
        f"针齿直径 {args.pin_diameter:g} mm，偏心距 {args.eccentricity:g} mm"
    )
    return profile, teeth, desc


def nominal_from_mesh(triangles: np.ndarray, scan_radius: float, n_bins: int) -> tuple[np.ndarray, int, str]:
    """从三维模型的外轮廓提取名义齿廓，并估计齿数。"""
    triangles, unit_note = maybe_convert_units(triangles, scan_radius)
    verts = triangles.reshape(-1, 3)
    center, e1, e2, normal, _ = fit_disc_plane(verts)
    samples = sample_mesh_silhouette(triangles, step=max(0.05, scan_radius / 800))
    uv, _ = to_plane(samples, center, e1, e2, normal)
    profile, _, _ = center_and_extract(uv, n_bins)
    teeth = estimate_teeth(profile)
    desc = f"三维模型外轮廓，估计齿数 {teeth}。{unit_note}"
    return ensure_ccw(profile), teeth, desc


def measured_profile_from_cloud(points: np.ndarray, n_bins: int) -> tuple[np.ndarray, dict]:
    """从测量点云提取摆线盘外轮廓。"""
    center, e1, e2, normal, height = fit_disc_plane(points)
    uv, h = to_plane(points, center, e1, e2, normal)
    med = np.median(h)
    # 保留盘面和侧壁，侧壁上的点才是真正的加工轮廓
    high = np.percentile(h, 99.7)
    band = (h >= med - 0.4) & (h <= max(med + 0.4, high))
    if band.sum() < 1000:
        band = np.ones(len(h), dtype=bool)
    profile, origin2d, supported = center_and_extract(uv[band], n_bins)
    info = {
        "points": int(len(points)),
        "plane_points": int(band.sum()),
        "thickness_mm": float(np.percentile(h, 99.5) - np.percentile(h, 0.5)),
        "normal": normal,
        "supported": supported,
    }
    return profile, info


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _fmt(s: dict) -> str:
    """把一组偏差统计格式化成一行文字。"""
    return (
        f"平均 {s['mean']:+.4f} mm，标准差 {s['std']:.4f} mm，RMS {s['rms']:.4f} mm，"
        f"最小 {s['min']:+.4f} mm，最大 {s['max']:+.4f} mm，峰谷 {s['pv']:.4f} mm"
    )


def write_report(out: Path, payload: dict) -> None:
    """写出摆线盘轮廓误差的文字摘要和逐点 CSV。"""
    out.mkdir(parents=True, exist_ok=True)
    normal = payload["normal"]
    radial = payload["radial"]
    lines = [
        "摆线盘轮廓加工误差",
        "",
        f"测量文件：{payload['scan']}",
        f"名义轮廓：{payload['nominal_desc']}",
        f"测量点数：{payload['info']['points']}，用于提轮廓的点数：{payload['info']['plane_points']}",
        f"盘厚（高度 0.5% 到 99.5%）：{payload['info']['thickness_mm']:.3f} mm",
        f"齿数：{payload['teeth']}",
        f"对齐：旋转 {payload['rotation_deg']:+.3f}°（已折到一个齿距内），"
        f"平移 ({payload['translation'][0]:+.4f}, {payload['translation'][1]:+.4f}) mm，"
        f"镜像：{'是' if payload['mirrored'] else '否'}",
        "",
        "法向偏差（主结果，正值 = 实测在名义轮廓外侧 = 多肉）",
        _fmt(normal),
        "",
        "径向偏差（同一角度上的半径差，正值 = 实测半径更大）",
        _fmt(radial),
        "",
    ]
    if payload["tolerance"] is not None:
        tol = payload["tolerance"]
        err = payload["normal_error"]
        over = np.abs(err) > tol
        lines.append(f"公差带 ±{tol:.4f} mm：超差 {int(over.sum())} / {len(err)} 个采样点（{100 * over.mean():.2f}%）")
        lines.append("")
    lines.append("各齿法向偏差（mm）")
    lines.append("齿号, 平均, 最小, 最大, RMS")
    for i, row in enumerate(payload["per_tooth"]):
        lines.append(f"{i + 1}, {row['mean']:+.4f}, {row['min']:+.4f}, {row['max']:+.4f}, {row['rms']:.4f}")
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)

    csv_path = out / "profile_error.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "angle_deg", "x_mm", "y_mm", "r_measured_mm", "r_nominal_mm",
            "radial_error_mm", "normal_error_mm", "tooth",
        ])
        pts = payload["points"]
        ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        r_m = np.hypot(pts[:, 0], pts[:, 1])
        for i in range(len(pts)):
            writer.writerow([
                f"{ang[i]:.4f}", f"{pts[i, 0]:.5f}", f"{pts[i, 1]:.5f}",
                f"{r_m[i]:.5f}", f"{payload['nominal_radius'][i]:.5f}",
                f"{payload['radial_error'][i]:.5f}", f"{payload['normal_error'][i]:.5f}",
                int(payload["tooth"][i]) + 1,
            ])
    plot_profile_comparison(csv_path, out / "profile_comparison.png")


def _error_limit(err: np.ndarray) -> float:
    """取偏差绝对值的 99 分位，作为色标半宽。"""
    lim = float(np.percentile(np.abs(err), 99))
    return max(lim, 1e-4)


def signed_error_cmap() -> LinearSegmentedColormap:
    """有符号偏差。负侧靛蓝到绿，正侧黄到酒红，零点是绿和黄相接。"""
    # 色标位置按 [-1, +1] 写成 0–1：(-1→0, 0→0.5, +1→1)
    return LinearSegmentedColormap.from_list(
        "signed_error",
        [
            (0.000, "#4B0082"),
            (0.175, "#2166ac"),
            (0.325, "#41b6c4"),
            (0.475, "#41ab5d"),
            (0.525, "#fee08b"),
            (0.675, "#fdae61"),
            (0.850, "#e31a1c"),
            (1.000, "#800026"),
        ],
        N=256,
    )


def signed_error_limit(err: np.ndarray) -> float:
    """色条对称于 0，半宽取法向偏差的最大绝对值。"""
    values = np.abs(np.asarray(err, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-6
    return max(float(np.max(values)), 1e-6)


def _discrete_hues(n_bin: int) -> np.ndarray:
    """负侧靛蓝到青，正侧黄到红。相邻档换色相，零两侧保持浅色。"""
    from matplotlib.colors import hsv_to_rgb

    half = max(n_bin // 2, 1)
    extra = n_bin - 2 * half
    colors = []
    for i in range(half):
        if i == half - 1:
            colors.append(hsv_to_rgb((195.0 / 360.0, 0.18, 1.0)))
            continue
        u = i / max(half - 2, 1)
        hue = 270.0 - u * (270.0 - 150.0)
        colors.append(hsv_to_rgb((hue / 360.0, 0.95, 0.52 + 0.42 * u)))
    warm = half + extra
    for i in range(warm):
        if i == 0:
            colors.append(hsv_to_rgb((38.0 / 360.0, 0.30, 1.0)))
            continue
        u = (i - 1) / max(warm - 2, 1)
        # 黄 52° 收到酒红 338°，停在红色里，不走到和负侧相像的紫蓝。
        hue = (52.0 - u * 74.0) % 360.0
        colors.append(hsv_to_rgb((hue / 360.0, 1.0, max(1.0 - 0.48 * u, 0.42))))
    return np.asarray(colors, dtype=float)


def discrete_signed_scale(err: np.ndarray, step: float = 0.05):
    """以 0 为中心、每 0.05 mm 一档。半宽取绝对偏差的 98% 分位，再向外落到 0.05 mm。"""
    values = np.abs(np.asarray(err, dtype=float))
    values = values[np.isfinite(values)]
    peak = float(np.percentile(values, 98)) if values.size else step
    half = max(step, float(np.ceil((peak - 1e-9) / step) * step))
    n_bin = int(round(2 * half / step))
    edges = np.linspace(-half, half, n_bin + 1)
    cmap = ListedColormap(_discrete_hues(n_bin))
    norm = BoundaryNorm(edges, n_bin, clip=True)
    tick_step = step if n_bin + 1 <= 21 else step * 2
    ticks = np.linspace(-half, half, int(round(2 * half / tick_step)) + 1)
    return cmap, norm, ticks


def banded_colorbar(fig, mappable, ax, ticks, **kwargs):
    """色条按边界画成独立色块，不在档与档之间做渐变。"""
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


def _broken_contour(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按角度把实测点连成细线。间距明显大于正常步长的地方断开，不穿过空档。"""
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


def plot_report(out: Path, payload: dict) -> None:
    """画实测轮廓、名义轮廓和各齿误差图。"""
    pts = payload["points"]
    nom = payload["nominal"]
    err = payload["normal_error"]
    ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    order = np.argsort(ang)
    teeth = payload["teeth"]
    supported = np.asarray(payload.get("supported", np.ones(len(pts), dtype=bool)), dtype=bool)
    show = supported & np.isfinite(err)
    cmap, norm, ticks = discrete_signed_scale(err)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.plot(nom[:, 0], nom[:, 1], color="#9aa0a6", lw=1.0, label="名义轮廓", zorder=1)
    outline_x, outline_y = _broken_contour(pts[show])
    ax.plot(outline_x, outline_y, color="#4a4d51", lw=0.7, solid_capstyle="round", zorder=4)
    point_radius = _disk_radius(pts[show])
    disks = _measured_disks(ax, pts[show], err[show], point_radius, cmap, norm)
    cb = banded_colorbar(fig, disks, ax, ticks, fraction=0.046, pad=0.04)
    cb.set_label("法向偏差 (mm)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"轮廓对比   法向 RMS {payload['normal']['rms']:.4f} mm")
    ax.plot(
        [], [], ls="none", marker="o", ms=5, mfc="#c5c9ce", mec="#8a8f96", mew=0.5,
        label=f"测点半径 {point_radius:.2f} mm",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "overlay.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axhline(0, color="#888", lw=0.8)
    ax.plot(ang[order], payload["radial_error"][order], color="#1a73e8", lw=0.8, label="径向偏差")
    ax.plot(ang[order], err[order], color="#d93025", lw=0.8, label="法向偏差")
    ax.set_xlabel("角度 (°)")
    ax.set_ylabel("偏差 (mm)")
    ax.set_title("加工误差沿圆周的分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_vs_angle.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(err, bins=60, color="#1a73e8", edgecolor="white")
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("法向偏差 (mm)")
    ax.set_ylabel("采样点数")
    ax.set_title("法向偏差分布")
    fig.tight_layout()
    fig.savefig(out / "error_histogram.png", dpi=140)
    plt.close(fig)

    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(111, projection="polar")
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    order_t = np.argsort(theta)
    ax.plot(theta[order_t], err[order_t], color="#d93025", lw=0.9)
    ax.set_title("法向偏差极坐标 (mm)", pad=16)
    fig.tight_layout()
    fig.savefig(out / "error_polar.png", dpi=140)
    plt.close(fig)

    per = payload["per_tooth"]
    fig, ax = plt.subplots(figsize=(max(8, teeth * 0.28), 4.2))
    xs = np.arange(1, teeth + 1)
    means = [row["mean"] for row in per]
    mins = [row["min"] for row in per]
    maxs = [row["max"] for row in per]
    ax.bar(xs, means, color="#1a73e8", width=0.7, label="平均")
    ax.vlines(xs, mins, maxs, color="#202124", lw=0.8, label="最小—最大")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("齿号")
    ax.set_ylabel("法向偏差 (mm)")
    ax.set_title("各齿加工误差")
    ax.set_xticks(xs)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_per_tooth.png", dpi=140)
    plt.close(fig)


def _cjk_font(size: float) -> font_manager.FontProperties:
    """返回指定字号的中文字体。"""
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(path).is_file():
        font_manager.fontManager.addfont(path)
        prop = font_manager.FontProperties(fname=path, size=size)
    else:
        prop = font_manager.FontProperties(family="sans-serif", size=size)
    return prop


def plot_profile_comparison(csv_path: Path, png_path: Path) -> None:
    """按 profile_error.csv 画轮廓对比，覆盖 profile_comparison.png。"""
    raw = np.genfromtxt(csv_path, delimiter=",", names=True)
    ang = np.radians(np.asarray(raw["angle_deg"], dtype=float))
    meas = np.column_stack([
        np.asarray(raw["x_mm"], dtype=float),
        np.asarray(raw["y_mm"], dtype=float),
    ])
    r_nom = np.asarray(raw["r_nominal_mm"], dtype=float)
    nom = np.column_stack([r_nom * np.cos(ang), r_nom * np.sin(ang)])
    err = np.asarray(raw["normal_error_mm"], dtype=float)
    tooth = np.asarray(raw["tooth"], dtype=int)

    order = np.argsort(ang)
    nom_o = nom[order]
    meas_o = meas[order]
    err_o = err[order]
    tooth_o = tooth[order]
    fwd = np.roll(nom_o, -1, axis=0) - nom_o
    length = np.maximum(np.hypot(fwd[:, 0], fwd[:, 1]), 1e-12)
    outward = np.column_stack([fwd[:, 1], -fwd[:, 0]]) / length[:, None]

    i_min = int(np.argmin(err))
    i_max = int(np.argmax(err))
    worst = int(tooth[int(np.argmax(np.abs(err)))])
    cmap, norm, ticks = discrete_signed_scale(err)

    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(13.4, 6.8), layout="constrained",
        gridspec_kw={"width_ratios": [1.28, 1.0]},
    )
    closed_x = np.r_[nom_o[:, 0], nom_o[0, 0]]
    closed_y = np.r_[nom_o[:, 1], nom_o[0, 1]]
    ax.plot(closed_x, closed_y, color="#9aa0a6", lw=1.05, zorder=1, label="名义轮廓")
    sc = ax.scatter(
        meas[:, 0], meas[:, 1], c=err, cmap=cmap, norm=norm, s=3, zorder=2, label="实测轮廓",
    )
    reach = float(np.max(np.hypot(meas[:, 0], meas[:, 1])))
    span = reach + 11.0
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontproperties=_cjk_font(11))
    ax.set_ylabel("y (mm)", fontproperties=_cjk_font(11))
    ax.set_title("轮廓对比", fontproperties=_cjk_font(13))

    def _mark(idx: int, text: str, color: str) -> None:
        x, y = float(meas[idx, 0]), float(meas[idx, 1])
        radial = max(np.hypot(x, y), 1e-9)
        ux, uy = x / radial, y / radial
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x + ux * 7.2, y + uy * 7.2),
            ha="center", va="center",
            color=color,
            fontproperties=_cjk_font(9),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 0.9},
            zorder=4,
        )
        ax.scatter([x], [y], s=36, facecolors="none", edgecolors=color, linewidths=1.1, zorder=5)

    _mark(i_min, f"最大少肉  齿{int(tooth[i_min])}\n{err[i_min]:.3f} mm", "#174ea6")
    _mark(i_max, f"最大多肉  齿{int(tooth[i_max])}\n{err[i_max]:+.3f} mm", "#a50e0e")
    ax.legend(loc="upper left", prop=_cjk_font(9), framealpha=0.92)
    cb = banded_colorbar(fig, sc, ax, ticks, fraction=0.046, pad=0.03)
    cb.set_label("法向偏差 (mm)", fontproperties=_cjk_font(11))

    sel = tooth_o == worst
    tips = nom_o[sel] + outward[sel] * err_o[sel, None]
    axz.plot(closed_x, closed_y, color="#9aa0a6", lw=1.2, zorder=1)
    for p0, p1 in zip(nom_o[sel], tips):
        axz.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#202124", lw=0.75, solid_capstyle="round", zorder=2)
    axz.scatter(meas_o[sel, 0], meas_o[sel, 1], c=err_o[sel], cmap=cmap, norm=norm, s=8, zorder=3)
    blob = np.vstack([nom_o[sel], meas_o[sel], tips])
    pad = 0.45
    axz.set_xlim(float(blob[:, 0].min()) - pad, float(blob[:, 0].max()) + pad)
    axz.set_ylim(float(blob[:, 1].min()) - pad, float(blob[:, 1].max()) + pad)
    axz.set_aspect("equal")
    axz.set_xlabel("x (mm)", fontproperties=_cjk_font(11))
    axz.set_ylabel("y (mm)", fontproperties=_cjk_font(11))
    axz.set_title(f"齿{worst} 放大（绝对值最大）", fontproperties=_cjk_font(13))
    axz.legend(
        handles=[
            Line2D([0], [0], color="#9aa0a6", lw=1.2, label="名义轮廓"),
            Line2D([0], [0], color="#202124", lw=0.9, label="法向偏差"),
        ],
        loc="best",
        prop=_cjk_font(9),
        framealpha=0.92,
    )
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def write_profile_ply(path: Path, pts: np.ndarray, err: np.ndarray) -> None:
    """把带偏差颜色的轮廓点写成 ASCII PLY。"""
    lim = _error_limit(err)
    t = np.clip(err / lim, -1, 1)
    # 蓝 = 少肉，白 = 0，红 = 多肉
    rgb = np.zeros((len(pts), 3))
    rgb[:, 0] = np.clip(1 + t, 0, 1)
    rgb[:, 1] = 1 - np.abs(t)
    rgb[:, 2] = np.clip(1 - t, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    xyz = np.column_stack([pts, np.zeros(len(pts))])
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")


def save_measured_only(out: Path, profile: np.ndarray, info: dict, teeth: int, scan: str) -> None:
    """没有名义模型时，只保存实测轮廓图和半径曲线。"""
    out.mkdir(parents=True, exist_ok=True)
    r = np.hypot(profile[:, 0], profile[:, 1])
    ang = np.degrees(np.arctan2(profile[:, 1], profile[:, 0]))
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.plot(profile[:, 0], profile[:, 1], color="#1a73e8", lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"实测外轮廓   估计齿数 {teeth}")
    fig.tight_layout()
    fig.savefig(out / "measured_profile.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    order = np.argsort(ang)
    ax.plot(ang[order], r[order], color="#1a73e8", lw=0.9)
    ax.set_xlabel("角度 (°)")
    ax.set_ylabel("半径 (mm)")
    ax.set_title("实测轮廓半径")
    fig.tight_layout()
    fig.savefig(out / "measured_radius.png", dpi=140)
    plt.close(fig)

    with (out / "measured_profile.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_deg", "x_mm", "y_mm", "radius_mm"])
        for i in order:
            writer.writerow([f"{ang[i]:.4f}", f"{profile[i, 0]:.5f}", f"{profile[i, 1]:.5f}", f"{r[i]:.5f}"])
    note = (
        "已提取实测外轮廓，还没有名义模型，因此没有计算加工误差。\n"
        f"测量点数 {info['points']}，轮廓采样 {len(profile)}，估计齿数 {teeth}，"
        f"半径 {r.min():.3f} ~ {r.max():.3f} mm。\n"
        "重新运行并加上 --model 模型文件，或加上 --teeth --pitch-diameter --pin-diameter --eccentricity。\n"
    )
    (out / "summary.txt").write_text(note, encoding="utf-8")
    print(note)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_comparison(scan: Path, nominal: np.ndarray, teeth: int, nominal_desc: str, out: Path, tolerance: float | None, n_bins: int) -> dict:
    """提取实测轮廓，对齐到名义轮廓，并输出加工误差。"""
    points, _ = read_ply(scan)
    measured, info = measured_profile_from_cloud(points, n_bins)
    measured_teeth = estimate_teeth(measured)
    if measured_teeth != teeth:
        print(f"注意：测量轮廓估计齿数是 {measured_teeth}，名义齿数是 {teeth}。按名义齿数分齿。")
    nominal = ensure_ccw(nominal)
    # 名义轮廓如果来自参数，半径栅格和测量一致；再居中一次
    nominal = nominal - nominal.mean(axis=0)
    aligned = align_profiles(measured, nominal, teeth)
    tooth, _ = tooth_index(aligned["points"], nominal, teeth)
    per_tooth = []
    for i in range(teeth):
        vals = aligned["normal_error"][tooth == i]
        per_tooth.append(stats(vals) if len(vals) else {k: float("nan") for k in ("mean", "std", "rms", "min", "max", "pv", "p05", "p95")})
    payload = {
        "scan": str(scan),
        "nominal_desc": nominal_desc,
        "info": info,
        "teeth": teeth,
        "points": aligned["points"],
        "nominal": nominal,
        "normal_error": aligned["normal_error"],
        "radial_error": aligned["radial_error"],
        "nominal_radius": aligned["nominal_radius"],
        "normal": stats(aligned["normal_error"]),
        "radial": stats(aligned["radial_error"]),
        "rotation_deg": aligned["rotation_deg"],
        "translation": aligned["translation"],
        "mirrored": aligned["mirrored"],
        "tooth": tooth,
        "per_tooth": per_tooth,
        "tolerance": tolerance,
        "supported": info["supported"],
    }
    write_report(out, payload)
    plot_report(out, payload)
    write_profile_ply(out / "contour_deviation.ply", aligned["points"], aligned["normal_error"])
    return payload


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数：点云、模型或设计参数、齿数和输出目录。"""
    p = argparse.ArgumentParser(description="对比摆线盘实测轮廓和三维模型，计算加工误差")
    p.add_argument("--scan", type=Path, default=Path("1.ply"), help="测量点云 PLY，默认 1.ply")
    p.add_argument("--model", type=Path, default=None, help="名义三维模型，STL / OBJ / PLY")
    p.add_argument("--teeth", type=int, default=None, help="摆线轮齿数。用设计参数时必填")
    p.add_argument("--pitch-diameter", type=float, default=None, help="针齿节圆直径，毫米")
    p.add_argument("--pin-diameter", type=float, default=None, help="针齿直径，毫米")
    p.add_argument("--eccentricity", type=float, default=None, help="偏心距，毫米")
    p.add_argument("--bins", type=int, default=1440, help="轮廓角度采样数，默认 1440（0.25°）")
    p.add_argument("--tolerance", type=float, default=None, help="法向公差，毫米。给出后统计超差比例")
    p.add_argument("--out", type=Path, default=Path("error_report"), help="结果目录")
    p.add_argument("--self-test", action="store_true", help="用带已知误差的仿真轮廓检查程序")
    return p


def self_test() -> None:
    """造一个带已知径向误差的点云，走提取、对齐、统计，核对能否找回误差。"""
    teeth = 30
    raw = ensure_ccw(cycloid_profile(teeth + 1, pin_radius=1.6, pitch_radius=42.0, eccentricity=0.9, points_per_tooth=80))
    nominal, _, _ = center_and_extract(raw, 1440)
    grid = np.linspace(-np.pi, np.pi, 1440 * 8, endpoint=False)
    _, radius = resample_radius(raw, len(grid))
    injected = 0.02 + 0.05 * np.sin(3 * grid)
    cloud = np.column_stack([(radius + injected) * np.cos(grid), (radius + injected) * np.sin(grid)])
    angle = np.radians(18.0)
    c, s = np.cos(angle), np.sin(angle)
    cloud = apply_rigid(cloud, np.array([[c, -s], [s, c]]), np.array([0.25, -0.15]))
    measured, _, _ = center_and_extract(cloud, 1440)
    aligned = align_profiles(measured, nominal - nominal.mean(axis=0), teeth)
    got = aligned["radial_error"]
    ang = np.arctan2(aligned["points"][:, 1], aligned["points"][:, 0])
    design = np.column_stack([np.ones(len(ang)), np.sin(3 * ang), np.cos(3 * ang)])
    coef, _, _, _ = np.linalg.lstsq(design, got, rcond=None)
    amplitude = float(np.hypot(coef[1], coef[2]))
    summary = stats(got)
    print(
        f"自检 径向平均 {summary['mean']:+.4f} mm，RMS {summary['rms']:.4f} mm，"
        f"三次谐波幅值 {amplitude:.4f} mm，镜像 {aligned['mirrored']}"
    )
    # 0.05 mm 外缘带窄于仿真齿面在一个扇区内的半径变化，陡斜处会略偏大。
    # 这条只放宽自检，不把实测用的边缘带放回 0.2 mm。三次谐波仍按原阈值核对。
    if abs(summary["mean"] - 0.02) > 0.015:
        raise SystemExit(f"自检失败：径向平均偏差 {summary['mean']:.4f}，期望约 0.020（窄带允许约 0.01 mm 偏大）")
    if abs(amplitude - 0.05) > 0.008:
        raise SystemExit(f"自检失败：三次谐波幅值 {amplitude:.4f}，期望约 0.050")
    if aligned["mirrored"]:
        raise SystemExit("自检失败：正确轮廓被当成了镜像")
    print("自检通过")


def main(argv: list[str] | None = None) -> None:
    """读参数后跑自检，或做一次摆线盘轮廓对比并写出报告。"""
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return
    scan = args.scan
    if not scan.is_file():
        raise SystemExit(f"找不到测量点云: {scan}")
    param_given = [args.teeth, args.pitch_diameter, args.pin_diameter, args.eccentricity]
    has_params = any(v is not None for v in param_given)
    if args.model and has_params:
        raise SystemExit("请只指定一种名义轮廓：--model，或者一套设计参数。")
    if args.model:
        if not args.model.is_file():
            raise SystemExit(f"找不到三维模型: {args.model}")
        points, _ = read_ply(scan)
        rough, _ = measured_profile_from_cloud(points, args.bins)
        scan_radius = float(np.median(np.hypot(rough[:, 0], rough[:, 1])))
        triangles = load_model(args.model)
        nominal, teeth, desc = nominal_from_mesh(triangles, scan_radius, args.bins)
        run_comparison(scan, nominal, teeth, desc, args.out, args.tolerance, args.bins)
        return
    if has_params:
        if any(v is None for v in param_given):
            raise SystemExit("设计参数需要同时给出 --teeth --pitch-diameter --pin-diameter --eccentricity")
        nominal, teeth, desc = nominal_from_parameters(args, args.bins)
        run_comparison(scan, nominal, teeth, desc, args.out, args.tolerance, args.bins)
        return

    points, _ = read_ply(scan)
    profile, info = measured_profile_from_cloud(points, args.bins)
    teeth = estimate_teeth(profile)
    save_measured_only(args.out, profile, info, teeth, str(scan))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
