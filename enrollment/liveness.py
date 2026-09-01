# ============================================
# SMART INVENTORY SYSTEM
# Module: Liveness Detection (Randomized Head-Pose Challenge)
#
# NEW 2026-08-31 — added as an anti-spoofing gate on the entrance
# camera. Before a detected face is trusted enough to match or enroll,
# the customer must complete a randomized sequence of head turns
# ("look left", "look up", ...). A static photo can't move, and
# because the sequence is freshly randomized each time, a pre-recorded
# video of the real person can't reliably match it either.
#
# Technique: MediaPipe Face Mesh landmarks -> cv2.solvePnP against a
# generic 6-point 3D face model -> yaw/pitch angles. Standard OpenCV
# head-pose-estimation approach, not something novel.
#
# UPDATED 2026-08-31 (real-camera feedback round) — first live test
# showed every attempt failing. Fixed based on that specific feedback:
#   - step_timeout was 6s with zero visual guidance -- raised to 12s,
#     plus a 1.5s "get ready" preamble before the countdown starts.
#   - Added a live on-screen readout ("Detected: LEFT/RIGHT/UP/DOWN/
#     CENTER") so it's visible in real time whether a turn is being
#     picked up at all, and in which direction -- this also means if
#     the yaw sign convention (see classify_pose below) IS backwards,
#     it'll show up immediately as "Detected: RIGHT" while the person
#     is turning left, instead of being an invisible black box.
#   - Added a face guide box (drawn from Face Mesh landmark bounds) and
#     a directional arrow showing which way to turn, addressing "no
#     bound box to guide customer".
#   - Attempt counting (3 tries before ending the session) is handled
#     by the caller (entrance.py), not here -- this function still just
#     reports pass/fail for one challenge.
#
# ⚠️ NOT VERIFIED AGAINST A REAL CAMERA. Still to confirm on next run:
#   1. YAW SIGN CONVENTION — watch the new "Detected: X" readout while
#      turning left. If it says RIGHT, flip the one line marked below
#      in classify_pose() — everything else is unaffected.
#   2. YAW_THRESHOLD / PITCH_THRESHOLD / HOLD_SECONDS — starting
#      values, not calibrated. If "Detected" never leaves CENTER even
#      on a full head turn, lower these.
# ============================================

import time
import random
import cv2
import numpy as np

# Generic 3D face model points (arbitrary units, consistent scale).
# Standard 6-point correspondence set used for solvePnP head pose.
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye, left corner
    (225.0, 170.0, -135.0),    # Right eye, right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0),   # Right mouth corner
], dtype="double")

# MediaPipe Face Mesh landmark indices matching the model points above,
# in the same order (nose tip, chin, L-eye, R-eye, L-mouth, R-mouth).
_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

YAW_THRESHOLD   = 15.0  # degrees past which a turn counts as "left"/"right"
PITCH_THRESHOLD = 12.0  # degrees past which a tilt counts as "up"/"down"
HOLD_SECONDS    = 0.4   # must hold the pose this long, not just flick through it

DIRECTIONS = ["left", "right", "up", "down"]

COL_WHITE = (255, 255, 255)
COL_AMBER = (0, 165, 255)
COL_GREEN = (0, 220, 0)
COL_RED   = (50, 50, 255)
COL_CYAN  = (255, 220, 0)


def random_challenge(n=3):
    """Random ordered sequence of n distinct directions, e.g. ['left', 'up', 'right']."""
    return random.sample(DIRECTIONS, k=min(n, len(DIRECTIONS)))


def _estimate_pose(frame, landmarks):
    """
    landmarks: result.multi_face_landmarks[0].landmark from MediaPipe Face Mesh.
    Returns (yaw, pitch) in degrees, or None if estimation fails.
    """
    h, w = frame.shape[:2]
    try:
        image_points = np.array([
            (landmarks[i].x * w, landmarks[i].y * h) for i in _LANDMARK_IDS
        ], dtype="double")
    except (IndexError, AttributeError):
        return None

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    ok, rotation_vec, _translation_vec = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, _roll = [float(a) for a in euler_angles.flatten()]

    # decomposeProjectionMatrix wraps pitch into [-180, 180]; normalize
    # so a small nod doesn't jump near +/-180.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return yaw, pitch


def classify_pose(yaw, pitch):
    """Buckets a (yaw, pitch) reading into 'left'/'right'/'up'/'down'/'center'."""
    if abs(yaw) >= YAW_THRESHOLD and abs(yaw) >= abs(pitch):
        # ⚠️ UNVERIFIED SIGN CONVENTION — if the on-screen "Detected:"
        # readout shows the opposite of the way the person is actually
        # turning, swap "right"/"left" on the line below.
        return "right" if yaw > 0 else "left"
    if abs(pitch) >= PITCH_THRESHOLD:
        return "down" if pitch > 0 else "up"
    return "center"


def _draw_direction_arrow(frame, direction, w, h, color):
    """Big arrow showing which way to turn, placed below the face area."""
    cx, cy = w // 2, min(h - 40, h * 2 // 3)
    length = 45
    if direction == "left":
        pt1, pt2 = (cx + length, cy), (cx - length, cy)
    elif direction == "right":
        pt1, pt2 = (cx - length, cy), (cx + length, cy)
    elif direction == "up":
        pt1, pt2 = (cx, cy + length), (cx, cy - length)
    else:  # down
        pt1, pt2 = (cx, cy - length), (cx, cy + length)
    cv2.arrowedLine(frame, pt1, pt2, color, 4, tipLength=0.35)


def run_liveness_check(cam, face_mesh, preprocess_fn, w, h, steps=3,
                        step_timeout=12.0, window_name="Camera 1 - Entrance"):
    """
    Runs an interactive randomized head-pose challenge on the already-
    connected `cam` (a CameraStream). `face_mesh` must be a persistent
    mediapipe.solutions.face_mesh.FaceMesh instance (created once by
    the caller, not per-call). `preprocess_fn` should match
    enrollment.py's _preprocess (resizes frames to w x h).

    Returns True if the customer completed the full randomized
    sequence within the time budget, False on timeout/failure/'Q'.

    Does NOT release the camera or destroy windows — the caller
    (entrance.py) owns that lifecycle. Also does not count attempts —
    the caller decides how many failures to allow before giving up.
    """
    # Brief "get ready" preamble so the challenge doesn't start the
    # instant the customer becomes stable — gives them a beat to
    # notice the prompt before the countdown begins.
    prep_until = time.time() + 1.5
    while time.time() < prep_until:
        frame = cam.read_frame()
        if frame is None:
            time.sleep(0.03)
            continue
        frame = preprocess_fn(frame)
        cv2.putText(frame, "Liveness check starting...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_CYAN, 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            return False

    challenge = random_challenge(steps)
    step_idx = 0
    step_started = time.time()
    hold_start = None

    while step_idx < len(challenge):
        frame = cam.read_frame()
        if frame is None:
            time.sleep(0.03)
            continue
        frame = preprocess_fn(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        current_pose = "center"
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            pose = _estimate_pose(frame, landmarks)
            if pose is not None:
                current_pose = classify_pose(*pose)

            # Face guide box, drawn from the landmark bounds -- gives
            # the customer visible confirmation they're being tracked.
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            box_color = COL_GREEN if current_pose != "center" else COL_AMBER
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        target = challenge[step_idx]
        elapsed_step = time.time() - step_started
        remaining = max(0.0, step_timeout - elapsed_step)

        if current_pose == target:
            if hold_start is None:
                hold_start = time.time()
            elif time.time() - hold_start >= HOLD_SECONDS:
                step_idx += 1
                step_started = time.time()
                hold_start = None
        else:
            hold_start = None

        color = COL_GREEN if current_pose == target else COL_AMBER
        cv2.putText(frame, f"Turn your head {target.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"Detected: {current_pose.upper()}   Time left: {remaining:0.0f}s",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1)
        cv2.putText(frame, f"Step {step_idx + 1}/{len(challenge)}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_WHITE, 1)

        _draw_direction_arrow(frame, target, w, h, color)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            return False

        if elapsed_step > step_timeout:
            cv2.putText(frame, "Liveness check timed out.", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_RED, 2)
            cv2.imshow(window_name, frame)
            cv2.waitKey(800)
            return False

    return True
