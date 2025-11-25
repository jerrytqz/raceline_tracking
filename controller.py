import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack


def controller(
    state: ArrayLike,
    parameters: ArrayLike,
    racetrack: RaceTrack
) -> ArrayLike:

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
    # 2. Compute current curvature at closest point
    # --------------------------------------------------------
    curvature_now = curvature_at(idx)

    # --------------------------------------------------------
    # 2b. Peak curvature in a short window ahead
    #     (captures "sharp + short straight + opposite" complexes)
    # --------------------------------------------------------
    peak_window_dist = 60.0   # meters of track to scan ahead (tune 40–80)
    dist_acc_peak = 0.0
    idx_scan = idx
    curvature_peak = curvature_now

    while dist_acc_peak < peak_window_dist:
        p1 = centerline[idx_scan % N]
        p2 = centerline[(idx_scan + 1) % N]
        seg_len = np.linalg.norm(p2 - p1)
        dist_acc_peak += seg_len

        idx_scan = (idx_scan + 1) % N
        kappa = curvature_at(idx_scan)
        if kappa > curvature_peak:
            curvature_peak = kappa

    # effective curvature used for planning:
    # use the worst curvature in the next ~60m so we respect the whole complex
    curvature_eff = curvature_peak

    # --------------------------------------------------------
    # 3. PURE PURSUIT with SPEED- and CURVATURE-DEPENDENT LOOKAHEAD
    #    using curvature_eff (not just curvature_now)
    # --------------------------------------------------------
    L0 = 3.0          # base lookahead [m] at v = 0
    k_v = 0.6         # how much lookahead grows with speed [s]

    Ld_speed = L0 + k_v * v
    Ld_speed = np.clip(Ld_speed, 3.0, 40.0)

    # PURE PURSUIT lookahead based only on speed
    L0 = 5.0          # slightly larger base
    k_v = 0.7

    Ld_speed = L0 + k_v * v

    # Don't let lookahead go below, say, 8 m
    lookahead_dist = np.clip(Ld_speed, 8.0, 40.0)


    # walk along centerline until we've gone lookahead_dist
    dist_acc = 0.0
    idx2 = idx

    while dist_acc < lookahead_dist:
        p1 = centerline[idx2 % N]
        p2 = centerline[(idx2 + 1) % N]
        dist_acc += np.linalg.norm(p2 - p1)
        idx2 += 1

    target = centerline[idx2 % N]

    # --------------------------------------------------------
    # 4. Determine desired heading to target
    # --------------------------------------------------------
    dx = target[0] - x
    dy = target[1] - y

    phi_desired = np.arctan2(dy, dx)
    alpha = phi_desired - phi

    # Wrap angle to [-pi, pi]
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

    # --------------------------------------------------------
    # 5. Pure Pursuit steering formula
    # --------------------------------------------------------
    delta_r = np.arctan2(2 * wheelbase * np.sin(alpha), lookahead_dist)

    # Limit steering angle
    delta_r = np.clip(delta_r, -parameters[4], parameters[4])

    # --------------------------------------------------------
    # 6. Curvature-based speed planning
    #    **also using curvature_eff**
    # --------------------------------------------------------
    v_max = 110.0      # top speed on straights
    k_speed = 8     # slows the car in turns

    # use the worst curvature in the upcoming window, not just local
    v_r = v_max / (1 + k_speed * curvature_eff)

    # --------------------------------------------------------
    # 7. Detect sudden curvature further ahead and slow down hard
    #    with SPEED-DEPENDENT LOOKAHEAD DISTANCE
    #    (kept as an additional safety net)
    # --------------------------------------------------------
    curvature_jump_factor = 2.2       # how much larger than current to be "sudden"
    curvature_abs_threshold = 0.10    # minimum absolute curvature for a "real" corner
    hard_slowdown_factor = 0.8        # how aggressively to slow

    hazard_min = 25.0   # m lookahead at very low speed
    hazard_max = 250.0  # m lookahead at top speed (or near it)

    v_ratio = np.clip(v / v_max, 0.0, 1.0)
    hazard_lookahead_dist = hazard_min + (hazard_max - hazard_min) * v_ratio

    max_curv_ahead = curvature_eff      # start from curvature_eff now
    dist_acc_hazard = 0.0
    idx_scan = idx

    while dist_acc_hazard < hazard_lookahead_dist:
        p1 = centerline[idx_scan % N]
        p2 = centerline[(idx_scan + 1) % N]
        seg_len = np.linalg.norm(p2 - p1)
        dist_acc_hazard += seg_len

        idx_scan = (idx_scan + 1) % N
        kappa = curvature_at(idx_scan)

        if kappa > max_curv_ahead:
            max_curv_ahead = kappa

    if (
        max_curv_ahead > curvature_abs_threshold
        and max_curv_ahead > curvature_jump_factor * curvature_eff
    ):
        v_corner = v_max / (1 + k_speed * max_curv_ahead)
        v_hazard = v_corner * hard_slowdown_factor
        v_r = min(v_r, v_hazard)

    # --------------------------------------------------------
    # 8. FINAL CLAMP with CURVATURE-DEPENDENT MIN SPEED
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

    # --------------------------------------------------------
    # Steering rate control
    # --------------------------------------------------------
    Kp_delta = 4.0
    steer_rate = Kp_delta * (delta_r - delta)

    steer_rate = np.clip(
        steer_rate,
        parameters[7],   # min steering rate
        parameters[9]    # max steering rate
    )

    # --------------------------------------------------------
    # Velocity control
    # --------------------------------------------------------
    Kp_v = 1.2
    accel = Kp_v * (v_r - v)

    accel = np.clip(
        accel,
        parameters[8],   # min accel (braking)
        parameters[10]   # max accel
    )

    return np.array([steer_rate, accel])
