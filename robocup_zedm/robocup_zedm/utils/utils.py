import numpy as np

def is_valid_depth(d: float) -> bool:
    return np.isfinite(d) and d > 0.0


def get_median_depth_m(depth_m: np.ndarray, u: int, v: int, half_win: int) -> float:
    """
    depth_m: HxW float32 meters
    u,v: pixel (col,row)
    half_win: window half size (e.g., 2 -> 5x5)
    return: median depth in meters, invalid -> -1.0
    """
    if depth_m is None or depth_m.size == 0:
        return -1.0
    if depth_m.dtype != np.float32 or depth_m.ndim != 2:
        return -1.0

    h, w = depth_m.shape[:2]
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))

    x0 = max(0, u - half_win)
    x1 = min(w - 1, u + half_win)
    y0 = max(0, v - half_win)
    y1 = min(h - 1, v + half_win)

    win = depth_m[y0:y1 + 1, x0:x1 + 1].reshape(-1)
    mask = np.isfinite(win) & (win > 0.0)
    vals = win[mask]
    if vals.size == 0:
        return -1.0

    mid = vals.size // 2
    return float(np.partition(vals, mid)[mid])


def pixel_to_cam_coords(depth_m: np.ndarray,
                        u: int, v: int,
                        fx: float, fy: float,
                        cx: float, cy: float,
                        half_win: int) -> tuple[float, float, float]:
    """
    returns (X,Y,Z) meters in camera frame.
    invalid -> (0,0,-1)
    """
    if fx <= 0.0 or fy <= 0.0:
        return (0.0, 0.0, -1.0)

    Z = get_median_depth_m(depth_m, u, v, half_win)
    if not is_valid_depth(Z):
        return (0.0, 0.0, -1.0)

    X = (float(u) - float(cx)) * Z / float(fx)
    Y = (float(v) - float(cy)) * Z / float(fy)
    return (X, Y, Z)

def rotation(pan_deg: int, tilt_deg: int, X: float, Y: float, Z: float):
    pan = np.deg2rad(float(pan_deg))
    tilt = np.deg2rad(float(tilt_deg))

    # Tilt: rotate around X axis
    c = np.cos(tilt); s = np.sin(tilt)
    R_tilt = np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c],
    ], dtype=np.float32)

    # Pan: rotate around Y axis
    c = np.cos(pan); s = np.sin(pan)
    R_pan = np.array([
        [  c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [ -s, 0.0,  c],
    ], dtype=np.float32)

    p = np.array([float(X), float(Y), float(Z)], dtype=np.float32)
    p2 = (R_pan @ (R_tilt @ p))   # tilt -> pan

    return (float(p2[0]), float(p2[1]), float(p2[2]))
