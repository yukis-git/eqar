#!/usr/bin/env python3
"""Draws docs/erp_detections.png: a synthetic 1920 x 960 equirectangular frame with markers
near the north pole, across the +-180 degree seam and near the south pole, and the corners
eqar finds on it (edges drawn as great-circle arcs). No real footage is used.

    python docs/make_figure.py            # needs eqar and numpy only
"""

import os
import sys
import zlib
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests", "py"))
import erp_synth as S  # noqa: E402

import eqar  # noqa: E402

W, H = 1920, 960
MARKERS = [
    dict(id=11, lon=-60, lat=18, half=8, roll=12),
    dict(id=23, lon=60, lat=-22, half=7, roll=-30),
    dict(id=5, lon=179, lat=8, half=9, roll=20),     # straddles the seam
    dict(id=31, lon=-130, lat=80, half=9, roll=35),  # near the north pole
    dict(id=42, lon=30, lat=-81, half=9, roll=-15),  # near the south pole
]


def scene():
    """Marker luminance frame plus a coloured backdrop (sky / ground gradient)."""
    gray = S.make_frame(W, MARKERS, bg=205, noise=0.0, blur=0.6)
    y = np.linspace(0, 1, H)[:, None, None]
    x = np.linspace(0, 2 * np.pi, W)[None, :, None]
    sky = np.array([205.0, 222.0, 240.0])  # R G B
    ground = np.array([226.0, 214.0, 190.0])
    back = sky * (1 - y) + ground * y
    back = back * (1 + 0.025 * np.sin(3 * x) * np.sin(6 * np.pi * y))
    # the marker pixels are the frame values; the quiet zone (205) takes the backdrop tint
    f = gray.astype(float)[:, :, None]
    tint = back / 205.0
    return np.clip(f * tint, 0, 255).astype(np.uint8)


def draw_dot(img, x, y, r, color):
    yy, xx = np.ogrid[-r: r + 1, -r: r + 1]
    m = xx * xx + yy * yy <= r * r
    for dy, dx in zip(*np.nonzero(m)):
        py, px = int(round(y)) + dy - r, int(round(x)) + dx - r
        if 0 <= py < H:
            img[py, px % W] = color


def draw_arc(img, a, b, color, n=60, width=1):
    """Great-circle arc from ray a to ray b, drawn in pixel space."""
    ang = np.arccos(np.clip(a @ b, -1, 1))
    t = np.linspace(0, 1, n)[:, None]
    p = (np.sin((1 - t) * ang) * a + np.sin(t * ang) * b) / np.sin(ang)
    for x, y in S.ray_to_pixel(p, W, H):
        draw_dot(img, x, y, width, color)


def write_png(path, rgb):
    h, w, _ = rgb.shape
    up = rgb.astype(np.int16)
    up[1:] -= rgb[:-1]  # PNG filter 2 (Up) compresses the smooth backdrop far better
    raw = b"".join(b"\x02" + (up[i] % 256).astype(np.uint8).tobytes() for i in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def main():
    img = scene()
    # no time limit: the default 2 ms budget can drop candidates on a busy machine
    det = eqar.Detector(W, H, time_budget_ms=None)
    found = det.detect(img, format="rgb")
    print("detected ids:", sorted(m.id for m in found))
    out = img.copy()
    teal, amber, red = (0, 150, 136), (255, 160, 0), (213, 0, 0)
    for m in found:
        r = m.rays.astype(float)
        for i in range(4):
            draw_arc(out, r[i], r[(i + 1) % 4], teal, width=1)
    for m in found:
        for i, (x, y) in enumerate(m.corners):
            draw_dot(out, x, y, 4 if i == 0 else 3, amber if i == 0 else red)
    path = os.path.join(HERE, "erp_detections.png")
    write_png(path, out)
    print(path, os.path.getsize(path) // 1024, "KB")


if __name__ == "__main__":
    main()
