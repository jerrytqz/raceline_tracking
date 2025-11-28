import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

def controller(
    state: ArrayLike,
    parameters: ArrayLike,
    racetrack: RaceTrack
) -> ArrayLike:
    """
    High-level controller:
    - Pure pursuit for steering.
    - Curvature-based speed planning using a speed-dependent arc-length window ahead.
    """

    # Set up and unpack variables
    x, y, delta, v, phi = state
    wheelbase = parameters[0]

    centerline = racetrack.centerline
    N = centerline.shape[0]

    car_pos = np.array([x, y])

    dists = np.linalg.norm(centerline - car_pos, axis=1)
    idx = np.argmin(dists)

    # STEERING -------------------------------------------------------------------------------------

    L0 = 5.0 # base lookahead
    k_v = 0.7 # lookahead growth with speed

    Ld_speed = L0 + k_v * v
    lookahead_dist = np.clip(Ld_speed, 8.0, 40.0)

    dist_acc = 0.0
    idx2 = idx
    while dist_acc < lookahead_dist:
        p1 = centerline[idx2 % N]
        p2 = centerline[(idx2 + 1) % N]
        dist_acc += np.linalg.norm(p2 - p1)
        idx2 += 1

    target = centerline[idx2 % N]

    dx = target[0] - x
    dy = target[1] - y

    phi_desired = np.arctan2(dy, dx)
    alpha = phi_desired - phi
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

    delta_r = np.arctan2(2 * wheelbase * np.sin(alpha), lookahead_dist)

    # VELOCITY -------------------------------------------------------------------------------------

    # Local curvature at a given centerline index
    def curvature_at(idx_local: int) -> float:
        i_prev = (idx_local - 2) % N
        i_curr = idx_local % N
        i_next = (idx_local + 2) % N

        p_prev = centerline[i_prev]
        p_curr = centerline[i_curr]
        p_next = centerline[i_next]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
        return float(np.abs(np.cross(v1, v2)) / denom)

    # Max curvature in a window of arc length ahead
    def max_curvature_ahead(start_idx: int, window_dist: float) -> float:
        dist_acc = 0.0
        idx_scan = start_idx
        curv_max = curvature_at(start_idx)

        while dist_acc < window_dist:
            p1 = centerline[idx_scan % N]
            p2 = centerline[(idx_scan + 1) % N]
            seg_len = np.linalg.norm(p2 - p1)
            dist_acc += seg_len

            idx_scan = (idx_scan + 1) % N
            kappa = curvature_at(idx_scan)
            if kappa > curv_max:
                curv_max = kappa

        return curv_max

    curvature_now = curvature_at(idx)

    v_max = 160.0
    v_ratio = np.clip(v / v_max, 0.0, 1.0)

    # Idea is that at low speed, we don't need to look very far but
    # at high speed, look much farther so we slow early for big corners
    window_min = 60.0
    window_max = 360.0
    speed_window_dist = window_min + (window_max - window_min) * v_ratio

    curvature_eff = max_curvature_ahead(idx, speed_window_dist)

    v_min = 5.0
    k_speed = 15.0
    v_r = v_min + (v_max - v_min) / (1.0 + k_speed * curvature_eff ** 1.4)

    return np.array([delta_r, v_r])

def lower_controller(
    state: ArrayLike,
    desired: ArrayLike,
    parameters: ArrayLike
) -> ArrayLike:
    """
    Low-level controller:
    - Simple proportional controllers for steering rate and acceleration.
    """
    delta, v = state[2], state[3]
    delta_r, v_r = desired

    Kp_delta = 4.0
    steer_rate = Kp_delta * (delta_r - delta)

    Kp_v = 1.2
    accel = Kp_v * (v_r - v)

    return np.array([steer_rate, accel])
