# 摆线减速机误差测量

用扫描点云测量摆线减速机零件的加工误差。长度单位是毫米，点云为 PLY，顶点含 x、y、z。

三项测量：

| 程序 | 测什么 | 默认输入 |
| --- | --- | --- |
| `pin_hole_perpendicularity.py` | 外壳上 31 个针齿孔相对端面法向的垂直度 | `外壳完整点云.ply` |
| `pin_profile_error.py` | 某一截面上针齿廓的节圆、针齿半径和齿廓偏差 | `外壳正面.ply` |
| `compare_machining_error.py` | 摆线盘外轮廓相对名义齿廓的加工误差 | `摆线轮.ply` |

检测原理、坐标建立和每一步计算写在 [说明.md](说明.md)。

## 环境

Python 3，依赖 NumPy、SciPy、Matplotlib。

```bash
pip install numpy scipy matplotlib
```

## 运行

```bash
python pin_hole_perpendicularity.py --scan 外壳完整点云.ply
python pin_profile_error.py --scan 外壳正面.ply --pins 31 --pitch-diameter 85 --pin-diameter 5
python compare_machining_error.py --scan 摆线轮.ply --teeth 30 --pitch-diameter 85 --pin-diameter 5 --eccentricity 1
```

`--bins` 默认 1440，圆周每 0.25° 一个轮廓点。垂直度程序的 `--step` 默认 0.8 mm，是沿高度分层的厚度。针齿廓和摆线轮廓程序可以用 `--self-test`，用已知误差的仿真轮廓核对计算。

## 这次点云的针齿孔结果

数据来自 `外壳完整点云.ply`，相对该点云自己的端面法向，不是摆线盘和针齿之间的装配夹角。

| 项目 | 结果 |
| --- | --- |
| 直壁高度差 | 12.0 mm（z −14.4 mm 到 −1.8 mm） |
| 针齿中心圆半径 | 42.522 mm |
| 针齿半径 | 2.359 mm |
| 整圈轴线倾斜 | 0.288°，径向幅度 0.060 mm，偏向约 130° |
| 整圈扭转 | −0.033°，切向 −0.025 mm |

图和数值报告在 `正面视角/`。针齿廓报告在 `pin_report/`、`外壳正面_report/`，摆线轮廓报告在 `error_report/`、`measured_profile/`。

## 目录

- `pin_hole_perpendicularity.py`：针齿孔垂直度
- `pin_profile_error.py`：针齿齿廓误差
- `compare_machining_error.py`：摆线盘轮廓误差
- `说明.md`：原理和计算过程
- `*.ply`：实测点云。`外壳完整点云.ply` 用于针齿孔，`摆线轮.ply` 用于摆线盘
