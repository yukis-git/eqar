"""Pure-numpy synthetic equirectangular frames with ArUco markers pasted on the sphere.

Used by the tests, the benchmark and the documentation figure. Conventions (the same as
the detector's): ray(lon, lat) = (cos lat cos lon, sin lat, cos lat sin lon); pixel
(x, y) of a W x H frame is lon = (x + .5) / W * 2 pi - pi, lat = pi / 2 - (y + .5) / H * pi.
A marker is a flat square on the tangent plane at its centre ray c, with columns along
ex = c x up (rolled by `roll_deg` in the plane) and rows going down along -ey.
"""

import numpy as np

import eqar


def ray(lon, lat):
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), np.sin(lat), cl * np.sin(lon)], axis=-1)


def pixel_rays(x, y, w, h):
    lon = (x + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (y + 0.5) / h * np.pi
    return ray(lon, lat)


def ray_to_pixel(r, w, h):
    r = np.asarray(r, float)
    n = np.linalg.norm(r, axis=-1)
    lon = np.arctan2(r[..., 2], r[..., 0])
    lat = np.arcsin(np.clip(r[..., 1] / n, -1, 1))
    return np.stack(
        [(lon + np.pi) / (2 * np.pi) * w - 0.5, (np.pi / 2 - lat) / np.pi * h - 0.5], axis=-1
    )


def frame_at(lon_deg, lat_deg, roll_deg=0.0):
    lo, la = np.radians(lon_deg), np.radians(lat_deg)
    c = ray(lo, la)
    e0 = np.cross(c, [0.0, 1.0, 0.0])
    n = np.linalg.norm(e0)
    e0 = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else e0 / n
    e1 = np.cross(e0, c)
    r = np.radians(roll_deg)
    ex = e0 * np.cos(r) + e1 * np.sin(r)
    ey = -e0 * np.sin(r) + e1 * np.cos(r)
    return c, ex, ey


def corner_rays(lon_deg, lat_deg, half_deg, roll_deg=0.0, dictionary="4x4_50"):
    """Exact TL, TR, BR, BL tag-corner rays (4, 3) of a marker made by `paste_marker`."""
    c, ex, ey = frame_at(lon_deg, lat_deg, roll_deg)
    h = np.tan(np.radians(half_deg))
    k = _k_tag(dictionary)
    sx = np.array([-1.0, 1.0, 1.0, -1.0])
    sy = np.array([1.0, 1.0, -1.0, -1.0])
    p = c + (sx * k * h)[:, None] * ex + (sy * k * h)[:, None] * ey
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def _k_tag(dictionary):
    ncell = int(dictionary[0]) + 2
    return ncell / (ncell + 3)


def paste_marker(img, marker_id, lon_deg, lat_deg, half_deg, roll_deg=0.0,
                 dictionary="4x4_50", dark=40.0, bright=220.0, ss=3):
    """Paints marker `marker_id` (dark `dark`, bright `bright`, 1.5-cell quiet zone of
    `bright`) into the float or uint8 gray frame `img`, in place, anti-aliased with an
    `ss` x `ss` supersample. `half_deg` is the half angle of the quiet-zone square."""
    hgt, wid = img.shape
    ncell = int(dictionary[0]) + 2
    grid = eqar.generate_marker(marker_id, dictionary, size=ncell) > 0
    k = _k_tag(dictionary)
    modw = 2 * k / ncell
    c, ex, ey = frame_at(lon_deg, lat_deg, roll_deg)
    h = np.tan(np.radians(half_deg))
    theta = np.degrees(np.arctan(np.sqrt(2) * h)) * 1.4 + 1.0
    lat_lo, lat_hi = max(lat_deg - theta, -89.999), min(lat_deg + theta, 89.999)
    row_of = lambda d: (0.5 * np.pi - np.radians(d)) / np.pi * hgt - 0.5
    r0, r1 = max(0, int(np.floor(row_of(lat_hi))) - 1), min(hgt - 1, int(np.ceil(row_of(lat_lo))) + 1)
    if lat_hi >= 89.5 or lat_lo <= -89.5:
        cols = np.arange(wid)
    else:
        cl = max(np.cos(np.radians(min(abs(lat_lo), abs(lat_hi)))), 0.05)
        m = theta / cl
        col_of = lambda d: (np.radians(d) + np.pi) / (2 * np.pi) * wid - 0.5
        c0 = int(np.floor(col_of(lon_deg - m))) - 1
        c1 = int(np.ceil(col_of(lon_deg + m))) + 1
        cols = np.arange(c0, c1 + 1) if c1 - c0 + 1 < wid else np.arange(wid)
    ys = np.arange(r0, r1 + 1)
    X, Y = np.meshgrid(cols, ys)
    acc = np.zeros(X.shape)
    hits = np.zeros(X.shape)
    for sy in range(ss):
        for sx in range(ss):
            fx = X + ((sx + 0.5) / ss - 0.5)
            fy = Y + ((sy + 0.5) / ss - 0.5)
            d = pixel_rays(fx, fy, wid, hgt)
            den = d @ c
            ok = den > 1e-12
            t = np.where(ok, 1.0 / np.where(ok, den, 1.0), 0.0)
            q = d * t[..., None] - c
            qx, qy = (q @ ex) / h, (q @ ey) / h
            inside = ok & (np.abs(qx) <= 1) & (np.abs(qy) <= 1)
            col = np.clip(np.floor((qx + k) / modw).astype(int), 0, ncell - 1)
            row = np.clip(np.floor((k - qy) / modw).astype(int), 0, ncell - 1)
            val = np.where(grid[row, col], bright, dark)
            val = np.where((np.abs(qx) > k) | (np.abs(qy) > k), bright, val)
            acc += np.where(inside, val, 0.0)
            hits += inside
    xi = X % wid
    old = img[Y, xi].astype(float)
    new = (acc + (ss * ss - hits) * old) / (ss * ss)
    img[Y, xi] = np.where(hits > 0, np.floor(new + 0.5), old).astype(img.dtype)
    return img


def board_frame(lon_deg, lat_deg, tilt_deg=0.0, yaw_deg=0.0):
    """(c, ex, ey, n) of a physical board centred on ray (lon, lat): standing upright and
    facing the camera, yawed by `yaw_deg` about the vertical, then tilted back by `tilt_deg`
    (top edge away from the camera). ex points right in the frame, ey up the board."""
    c = ray(np.radians(lon_deg), np.radians(lat_deg))
    up = np.array([0.0, 1.0, 0.0])
    hz = c - c[1] * up
    hz /= np.linalg.norm(hz)  # horizontal, away from the camera
    ex = np.cross(hz, up)
    ex /= np.linalg.norm(ex)
    yw = np.radians(yaw_deg)
    ex, hz = ex * np.cos(yw) + hz * np.sin(yw), -ex * np.sin(yw) + hz * np.cos(yw)
    t = np.radians(tilt_deg)
    ey = np.cos(t) * up + np.sin(t) * hz
    return c, ex, ey, np.cross(ex, ey)


def board_corner_rays(lon_deg, lat_deg, dist, side=0.42, tilt_deg=0.0, yaw_deg=0.0):
    """Exact TL, TR, BR, BL tag-corner rays (4, 3) of a marker made by `paste_board`."""
    c, ex, ey, _ = board_frame(lon_deg, lat_deg, tilt_deg, yaw_deg)
    h = side / 2
    p = dist * c + np.array([[-h, h], [h, h], [h, -h], [-h, -h]]) @ np.stack([ex, ey])
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def paste_board(img, marker_id, lon_deg, lat_deg, dist, side=0.42, board=None, tilt_deg=0.0,
                yaw_deg=0.0, dictionary="4x4_50", dark=40.0, bright=220.0, ss=3):
    """Paints a physical planar board at `dist` m along ray (lon, lat), oriented as in
    `board_frame`: a `board` m wide `bright` square (default: the 1.5-cell quiet zone only)
    carrying marker `marker_id` whose black square is `side` m. Perspective-exact (unlike
    `paste_marker`, which paints on the tangent plane of the unit sphere), anti-aliased with
    an `ss` x `ss` supersample, in place."""
    hgt, wid = img.shape
    ncell = int(dictionary[0]) + 2
    grid = eqar.generate_marker(marker_id, dictionary, size=ncell) > 0
    cell = side / ncell
    hb = side / 2 + 1.5 * cell if board is None else board / 2
    c, ex, ey, n = board_frame(lon_deg, lat_deg, tilt_deg, yaw_deg)
    p0 = dist * c
    crn = np.array([p0 + a * hb * ex + b * hb * ey for a in (-1, 1) for b in (-1, 1)])
    px = ray_to_pixel(crn, wid, hgt)
    if px[:, 0].max() - px[:, 0].min() > wid / 2:
        raise ValueError("paste_board: boards across the +-180 seam are not supported")
    x0, x1 = int(px[:, 0].min()) - 3, int(px[:, 0].max()) + 4
    y0, y1 = max(0, int(px[:, 1].min()) - 3), min(hgt - 1, int(px[:, 1].max()) + 4)
    X, Y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    acc = np.zeros(X.shape)
    hits = np.zeros(X.shape)
    pn = p0 @ n
    for sy in range(ss):
        for sx in range(ss):
            d = pixel_rays(X + ((sx + 0.5) / ss - 0.5), Y + ((sy + 0.5) / ss - 0.5), wid, hgt)
            den = d @ n
            ok = np.abs(den) > 1e-12
            t = np.where(ok, pn / np.where(ok, den, 1.0), -1.0)
            q = d * t[..., None] - p0
            u, v = q @ ex, q @ ey
            inside = ok & (t > 0) & (np.abs(u) <= hb) & (np.abs(v) <= hb)
            col = np.clip(np.floor((u + side / 2) / cell).astype(int), 0, ncell - 1)
            row = np.clip(np.floor((side / 2 - v) / cell).astype(int), 0, ncell - 1)
            val = np.where(grid[row, col], bright, dark)
            val = np.where((np.abs(u) > side / 2) | (np.abs(v) > side / 2), bright, val)
            acc += np.where(inside, val, 0.0)
            hits += inside
    xi = X % wid
    old = img[Y, xi].astype(float)
    new = (acc + (ss * ss - hits) * old) / (ss * ss)
    img[Y, xi] = np.where(hits > 0, np.floor(new + 0.5), old).astype(img.dtype)
    return img


def make_frame(w, markers, bg=205, noise=0.0, seed=0, blur=0.0, boards=()):
    """uint8 (w/2, w) gray frame. `markers` = iterable of dicts for `paste_marker`
    (keys id, lon, lat, half, roll); `boards` = iterable of keyword dicts for `paste_board`
    (painted after the markers)."""
    img = np.full((w // 2, w), float(bg))
    for m in markers:
        paste_marker(img, m["id"], m["lon"], m["lat"], m["half"], m.get("roll", 0.0))
    for b in boards:
        paste_board(img, **b)
    if blur > 0:
        r = int(np.ceil(3 * blur))
        kx = np.exp(-0.5 * (np.arange(-r, r + 1) / blur) ** 2)
        kx /= kx.sum()
        img = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="wrap"), kx, "valid"), 1, img)
        img = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), kx, "valid"), 0, img)
    if noise > 0:
        img = img + np.random.default_rng(seed).normal(0.0, noise, img.shape)
    return np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8)


def angular_error_px(rays_a, rays_b, w):
    """Angle between matching rays in equivalent pixels (2 pi / w rad per pixel)."""
    a = np.asarray(rays_a, float)
    b = np.asarray(rays_b, float)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.arctan2(cross, dot) / (2 * np.pi / w)
