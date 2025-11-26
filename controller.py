import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack


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
    - Curvature-based speed planning using a speed-dependent
      arc-length window ahead (single pass, no extra hazard block).
    """

    # Unpack the car state
    x, y, delta, v, phi = state
    wheelbase = parameters[0]

    centerline = racetrack.centerline
    N = centerline.shape[0]

    car_pos = np.array([x, y])

    # --------------------------------------------------------
    # 1. Find the closest point on the track
    # --------------------------------------------------------
    dists = np.linalg.norm(centerline - car_pos, axis=1)
    idx = np.argmin(dists)

    # --------------------------------------------------------
    # Helper: local curvature at a given centerline index
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Helper: max curvature in a window of arc length ahead
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 2. Curvature now and effective curvature ahead
    #    (speed-dependent window, replaces the hazard block)
    # --------------------------------------------------------
    curvature_now = curvature_at(idx)

    v_max = 100.0  # same as in your speed plan
    v_ratio = np.clip(v / v_max, 0.0, 1.0)

    # at low speed, we don't need to look very far;
    # at high speed, look much farther so we slow early for big corners
    window_min = 60.0    # m, similar to your original peak_window_dist
    window_max = 240.0   # m, similar scale to previous hazard_lookahead max
    speed_window_dist = window_min + (window_max - window_min) * v_ratio

    curvature_eff = max_curvature_ahead(idx, speed_window_dist)

    # --------------------------------------------------------
    # 3. SPEED-DEPENDENT LOOKAHEAD (pure pursuit)
    # --------------------------------------------------------
    L0 = 5.0      # base lookahead [m]
    k_v = 0.7     # lookahead growth with speed

    Ld_speed = L0 + k_v * v
    lookahead_dist = np.clip(Ld_speed, 8.0, 40.0)

    # Walk along centerline until we've gone lookahead_dist
    dist_acc = 0.0
    idx2 = idx
    while dist_acc < lookahead_dist:
        p1 = centerline[idx2 % N]
        p2 = centerline[(idx2 + 1) % N]
        dist_acc += np.linalg.norm(p2 - p1)
        idx2 += 1

    target = centerline[idx2 % N]

    # --------------------------------------------------------
    # 4. Desired heading to target
    # --------------------------------------------------------
    dx = target[0] - x
    dy = target[1] - y

    phi_desired = np.arctan2(dy, dx)
    alpha = phi_desired - phi
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # wrap to [-pi, pi]

    # --------------------------------------------------------
    # 5. Pure Pursuit steering
    # --------------------------------------------------------
    delta_r = np.arctan2(2 * wheelbase * np.sin(alpha), lookahead_dist)

    # --------------------------------------------------------
    # 6. Curvature-based speed planning (using curvature_eff)
    # --------------------------------------------------------
    k_speed = 8.0   # slows the car in turns

    v_r = v_max / (1 + k_speed * curvature_eff)

    # --------------------------------------------------------
    # 7. FINAL SPEED CLAMP with curvature-dependent min speed
    # --------------------------------------------------------
    v_min_base = 8.0       # m/s ~ 29 km/h
    sharp_curv_threshold = 0.25
    v_min_sharp = 4.0      # m/s ~ 14 km/h

    if curvature_now > sharp_curv_threshold:
        v_min = v_min_sharp
    else:
        v_min = v_min_base

    v_r = np.clip(v_r, v_min, v_max)

    return np.array([delta_r, v_r])

# ============================================================
# LOW-LEVEL CONTROLLER: Converts desired steering/speed → inputs
#   (unchanged)
# ============================================================

def lower_controller(
    state: ArrayLike,
    desired: ArrayLike,
    parameters: ArrayLike
) -> ArrayLike:
    delta, v = state[2], state[3]
    delta_r, v_r = desired

    Kp_delta = 4.0
    steer_rate = Kp_delta * (delta_r - delta)

    Kp_v = 1.2
    accel = Kp_v * (v_r - v)

    return np.array([steer_rate, accel])
