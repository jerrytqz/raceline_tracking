import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack


# ============================================================
# HIGH-LEVEL CONTROLLER: Computes desired steering + speed
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
    # 2. Pure Pursuit lookahead — stable and simple
    # --------------------------------------------------------
    lookahead_dist = 20.0   # meters
    dist_acc = 0.0
    idx2 = idx

    while dist_acc < lookahead_dist:
        p1 = centerline[idx2 % N]
        p2 = centerline[(idx2 + 1) % N]
        dist_acc += np.linalg.norm(p2 - p1)
        idx2 += 1

    target = centerline[idx2 % N]

    # --------------------------------------------------------
    # 3. Determine desired heading to target
    # --------------------------------------------------------
    dx = target[0] - x
    dy = target[1] - y

    phi_desired = np.arctan2(dy, dx)
    alpha = phi_desired - phi

    # Wrap angle to [-pi, pi]
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

    # --------------------------------------------------------
    # 4. Pure Pursuit steering formula
    # --------------------------------------------------------
    delta_r = np.arctan2(2 * wheelbase * np.sin(alpha), lookahead_dist)

    # Limit steering angle
    delta_r = np.clip(delta_r, -parameters[4], parameters[4])

    # --------------------------------------------------------
    # 5. Curvature-based speed planning
    # --------------------------------------------------------
    idx_prev = (idx - 2) % N
    idx_next = (idx + 2) % N

    p_prev = centerline[idx_prev]
    p_curr = centerline[idx]
    p_next = centerline[idx_next]

    v1 = p_curr - p_prev
    v2 = p_next - p_curr

    curvature = np.abs(
        np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    )

    # Speed model
    v_max = 45.0      # top speed on straights
    k_speed = 15.0    # slows the car in turns

    v_r = v_max / (1 + k_speed * curvature)
    v_r = np.clip(v_r, 12.0, v_max)

    return np.array([delta_r, v_r])


# ============================================================
# LOW-LEVEL CONTROLLER: Converts desired steering/speed → inputs
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
