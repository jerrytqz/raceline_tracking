import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack


# ============================================================
# HIGH-LEVEL CONTROLLER: Computes desired steering + speed
#   - pure pursuit
#   - curvature-based speed
#   - sudden-curvature slowdown
#   - NEW: curvature-dependent lookahead
# ============================================================

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

    # --------------------------------------------------------
    # 1. Find the closest point on the track
    # --------------------------------------------------------
    dists = np.linalg.norm(centerline - np.array([x, y]), axis=1)
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
    # 3. PURE PURSUIT with CURVATURE-DEPENDENT LOOKAHEAD
    # --------------------------------------------------------
    # idea:
    #   - in tight corners (high curvature) → shorter lookahead
    #   - on straights / after turns (low curvature) → longer lookahead
    lookahead_min = 12.0   # meters, for sharp turns
    lookahead_max = 35.0   # meters, on straights

    # scale curvature into [0, 1] for blending
    # curv_scale is a "typical sharp turn" curvature; tune as needed
    curv_scale = 0.15
    c_norm = np.clip(curvature_now / curv_scale, 0.0, 1.0)

    # when c_norm = 0 (straight) → lookahead_max
    # when c_norm = 1 (sharp)   → lookahead_min
    lookahead_dist = lookahead_max - (lookahead_max - lookahead_min) * c_norm

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
    # 6. Curvature-based speed planning (current point)
    # --------------------------------------------------------
    v_max = 105.0      # top speed on straights
    k_speed = 15.0    # slows the car in turns

    v_r = v_max / (1 + k_speed * curvature_now)

    # --------------------------------------------------------
    # 7. Detect sudden curvature ahead and slow down hard
    # --------------------------------------------------------
    hazard_steps_ahead = 30           # how many indices ahead to inspect
    curvature_jump_factor = 1.8       # how much larger than current to be "sudden"
    curvature_abs_threshold = 0.08    # minimum absolute curvature for a "real" corner
    hard_slowdown_factor = 0.45       # how aggressively to slow

    max_curv_ahead = curvature_now
    idx_scan = idx

    for _ in range(hazard_steps_ahead):
        idx_scan = (idx_scan + 1) % N
        kappa = curvature_at(idx_scan)
        if kappa > max_curv_ahead:
            max_curv_ahead = kappa

    if (
        max_curv_ahead > curvature_abs_threshold and
        max_curv_ahead > curvature_jump_factor * curvature_now
    ):
        v_corner = v_max / (1 + k_speed * max_curv_ahead)
        v_hazard = v_corner * hard_slowdown_factor
        v_r = min(v_r, v_hazard)

    # final clamp
    v_r = np.clip(v_r, 12.0, v_max)

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
