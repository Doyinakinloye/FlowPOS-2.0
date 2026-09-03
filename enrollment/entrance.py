# ============================================
# SMART INVENTORY SYSTEM
# Module: Entrance Engine (Camera 1)
# Description: Continuous recognize-or-enroll engine, with pause check
#
# REPLACED 2026-08-31 — this used to be a placeholder with a literal
# "connect your recognition model here" comment. Built from pieces
# that already existed elsewhere in the project: MediaPipe face
# detection + ArcFace embeddings (enrollment.py's helper functions),
# embedding_loader.find_match() for matching against the DB,
# enrollment.run_enrollment() for new customers, and
# face_db.get_most_recent_customer() — which was ALREADY written with
# a docstring saying "Used by entrance.py... so it can open their
# shopping session immediately." That confirmed this design rather
# than me guessing at it.
#
# UPDATED same day — added, then REMOVED, a liveness gate here.
#
# REWORKED 2026-08-31 (entry flow rework) — at the user's explicit
# request:
#   1. Liveness moved OUT of this file entirely — it now only runs
#      during enrollment (enrollment/enrollment.py), not for every
#      returning customer. This file no longer creates a Face Mesh
#      model or calls run_liveness_check at all; recognition is a
#      straight detect -> center-check -> stable -> match flow.
#   2. The old GUIDE_BOX_MIN_PX/MAX_PX width check (a distance proxy)
#      is replaced with the same box/centering check now used in
#      enrollment.py (imported from there, not duplicated) — accepts
#      a face at any distance as long as it's centered in the guide
#      box, and prompts with a direction when it isn't.
#   3. Welcome messages now say "Welcome Customer {ID}" instead of a
#      name, matching enrollment.py no longer collecting names.
#   4. New-customer detection now shows a brief on-screen "New
#      Customer Detected" notice before the camera handoff to
#      enrollment, instead of switching windows abruptly.
#
# ⚠️ STILL UNVERIFIED: guide-box left/right hints assume the preview
# isn't mirrored (see enrollment.py's _face_position_check for the
# same caveat) — and all timing constants below are starting values,
# not calibrated against a real camera.
#
# UPDATED 2026-09-01 — run_enrollment() now returns "success" /
# "flagged" / "cancelled" instead of a bool (enrollment.py added real
# duplicate detection). A "flagged" result shows a distinct on-screen
# message and issues no PIN — the customer must be resolved via
# Admin Panel > Review Flagged Enrollment before they can get one.
#
# FIXED 2026-09-01 (real-camera crash) — same bug as enrollment.py:
# `_mediapipe_face_crop(...) or upscaled` crashes with "The truth
# value of an array with more than one element is ambiguous" the
# moment the secondary detection actually SUCCEEDS (returns a real
# array, not None) -- which is exactly why the entrance camera was
# flashing on and immediately going dark in a loop: connect, crash on
# the first real detection, get caught by main.py's retry handler,
# reconnect, repeat. That retry loop was also why the popup felt slow
# to appear -- DeepFace's TensorFlow backend was reloading from
# scratch on every single retry. Fixed with an explicit `is None`
# check; verified by re-running the full loop with a mocked detection
# that always succeeds (the exact previously-crashing case) end to
# end with no exception.
#
# FIXED 2026-09-01 (later same day) — 'Q' and main.py's "Stop
# Entrance Engine" menu option both silently failed to actually stop
# anything; only Ctrl+C (killing the whole process) worked. Root
# cause: pressing 'Q' only `break`s this file's own while loop -- it
# never set this module's ENTRANCE_RUNNING flag, so main.py's outer
# daemon loop had no way to tell "this ended on purpose" from "this
# ended, try again" and just kept reconnecting. Separately, main.py's
# stop_entrance() only touched ITS OWN disconnected flag, never this
# module's -- so a genuinely running/blocked camera loop never even
# heard about it. Fixed here by having 'Q' explicitly set
# ENTRANCE_RUNNING = False (not just break); main.py was updated to
# check that flag and to actually call stop_entrance_engine(). Also
# made the "Could not reconnect after enrollment. Stopping." path
# actually stop, since it said so but didn't before. Verified three
# ways with real threads, not just reasoning: (1) a simulated 'Q'
# press no longer causes a retry, (2) stop_entrance() now interrupts
# a genuinely blocked/running fake loop, (3) a real connection-failure
# case still retries exactly as before -- no regression there.
#
# Camera handling: this loop owns Camera 1 via CameraStream. When an
# unrecognized face is matched to no one, it releases the camera and
# hands off to run_enrollment() (which opens its own connection) — the
# two can't hold the same camera at once — then reconnects afterward.
# ============================================

import os
import sys
import cv2
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_stream import CameraStream
from embedding_loader import find_match
from enrollment.enrollment import (
    run_enrollment, _preprocess, _upscale_bbox_crop,
    _mediapipe_face_crop, _clahe_preprocess, _extract_embedding,
    _face_position_check, _draw_guide_box,
    H, W,
)
from face_db import get_most_recent_customer
from services.shopping_session import (
    create_session,
    is_entrance_paused
)

# Global control flag for the entrance loop
ENTRANCE_RUNNING = True

# Optional hook, set via set_event_hook(). Fires exactly once per
# issued PIN (returning-customer match OR fresh enrollment), with
# {"is_new": bool, "name": str, "staff_id": str, "pin": str}. Does
# nothing by default -- purely additive, zero effect on the terminal
# flow unless something (sis_server.py) explicitly registers a hook.
# NOT a return-value change: start_entrance_engine() loops internally
# forever by design, so returning per-event would make main.py's
# outer loop reconnect the camera after every single customer.
_ON_EVENT = None


def set_event_hook(fn):
    """Register fn(event_dict) to be called on every match/enrollment
    PIN issuance. Pass None to clear it."""
    global _ON_EVENT
    _ON_EVENT = fn

RECOGNIZE_STABLE_SECONDS = 1.0  # face must hold steady (centered) this long before matching
MAX_RECOGNITION_FAILURES = 3    # consecutive failed embedding extractions before a visible failure message
RECOGNITION_FAIL_COOLDOWN = 3.0 # brief pause after MAX_RECOGNITION_FAILURES, before re-arming
COOLDOWN_SECONDS         = 8.0  # how long a success message (incl. the PIN) stays on screen before re-arming
NEW_CUSTOMER_NOTICE_SEC  = 1.5  # how long "New Customer Detected" shows before handing off to enrollment

COL_GREEN  = (0, 220, 0)
COL_BLACK  = (0, 0, 0)  # used for the Welcome/PIN success text specifically -- green washed out against some backgrounds
COL_YELLOW = (0, 220, 255)
COL_GRAY   = (140, 140, 140)
COL_WHITE  = (255, 255, 255)
COL_CYAN   = (255, 220, 0)
COL_RED    = (50, 50, 255)


def stop_entrance_engine():
    """Stops the entrance engine loop."""
    global ENTRANCE_RUNNING
    ENTRANCE_RUNNING = False


def _resolve_source(camera_index):
    """None/empty -> webcam 0 (matches get_stream_url()'s documented
    contract: 'Returns the full stream URL, or None for webcam').
    Digit strings -> int. Anything else (a URL) passes through."""
    if camera_index is None or (isinstance(camera_index, str) and camera_index.strip() == ""):
        return 0
    if isinstance(camera_index, str) and camera_index.strip().lstrip("-").isdigit():
        return int(camera_index.strip())
    return camera_index


def _draw_cooldown_message(frame, lines, color):
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 28


def start_entrance_engine(camera_index=0):
    """
    Runs Camera 1 in a loop: detects a face, tries to match it against
    enrolled customers, issues a shopping PIN if recognized, or hands
    off to enrollment if not — then re-arms for the next person.
    Respects is_entrance_paused() to stay silent during checkout.
    """
    global ENTRANCE_RUNNING
    ENTRANCE_RUNNING = True

    print("\n" + "=" * 56)
    print("   ENTRANCE ENGINE (Camera 1) ACTIVE")
    print("   Press 'Q' inside the camera window to pause/stop.")
    print("=" * 56)

    source = _resolve_source(camera_index)

    cam = CameraStream(stream_url=source)
    if not cam.connect():
        print(f"❌ Could not connect to Camera 1 (Entrance Camera). Source: {source!r}")
        return

    mp_model = None
    try:
        import mediapipe as mp
        mp_model = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        stable_since = None
        cooldown_until = 0.0
        cooldown_lines = ["Thank you! Re-arming for next customer..."]
        cooldown_color = COL_CYAN
        recognition_failures = 0

        while ENTRANCE_RUNNING:
            if is_entrance_paused():
                time.sleep(0.5)
                continue

            frame = cam.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            frame = _preprocess(frame)

            if time.time() < cooldown_until:
                _draw_cooldown_message(frame, cooldown_lines, cooldown_color)
                cv2.imshow("Camera 1 - Entrance", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    print("\n  ⏸️ Entrance stream stopped by user.")
                    ENTRANCE_RUNNING = False
                    break
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mp_model.process(rgb)
            faces = []
            if result.detections:
                for det in result.detections:
                    bb = det.location_data.relative_bounding_box
                    x = int(bb.xmin * W)
                    y = int(bb.ymin * H)
                    w = int(bb.width * W)
                    h = int(bb.height * H)
                    faces.append((x, y, w, h))

            guide_col = COL_GRAY
            msg = "Step up to Camera 1..."
            box_color = COL_WHITE

            if len(faces) == 1:
                bbox = faces[0]
                centered, hint = _face_position_check(bbox)
                box_color = COL_GREEN if centered else COL_WHITE

                if centered:
                    if stable_since is None:
                        stable_since = time.time()
                    elapsed = time.time() - stable_since

                    if elapsed < RECOGNIZE_STABLE_SECONDS:
                        guide_col = COL_YELLOW
                        msg = f"Hold steady... {elapsed:.1f}s"
                    else:
                        upscaled = _upscale_bbox_crop(frame, bbox)
                        face_crop = None
                        if upscaled is not None:
                            face_crop = _mediapipe_face_crop(upscaled, mp_model)
                            if face_crop is None:
                                face_crop = upscaled
                            face_crop = _clahe_preprocess(face_crop)

                        t0 = time.time()
                        emb = _extract_embedding(face_crop) if face_crop is not None else None
                        print(f"  [timing] embedding extraction took {time.time() - t0:.2f}s")

                        if emb is not None:
                            recognition_failures = 0
                            match = find_match(emb)
                            if match:
                                pin = create_session(match["staff_id"], match["name"])
                                print(f"\n✅ Welcome Customer {match['staff_id']}! Session PIN: {pin}")
                                if _ON_EVENT:
                                    _ON_EVENT({"is_new": False, "name": match["name"],
                                               "staff_id": match["staff_id"], "pin": pin})
                                cooldown_lines = [
                                    f"Welcome Customer {match['staff_id']}!",
                                    f"Your checkout PIN: {pin}",
                                    "(valid for this shopping session only)",
                                ]
                                cooldown_color = COL_BLACK
                                cooldown_until = time.time() + COOLDOWN_SECONDS
                                stable_since = None
                            else:
                                print("\n👤 New customer detected — opening enrollment...")
                                notice_until = time.time() + NEW_CUSTOMER_NOTICE_SEC
                                while time.time() < notice_until:
                                    nframe = cam.read_frame()
                                    if nframe is None:
                                        break
                                    nframe = _preprocess(nframe)
                                    cv2.putText(nframe, "New Customer Detected", (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_CYAN, 2)
                                    cv2.putText(nframe, "Starting enrollment...", (10, 58),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GRAY, 1)
                                    cv2.imshow("Camera 1 - Entrance", nframe)
                                    if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q')):
                                        break

                                cam.release()
                                cv2.destroyAllWindows()

                                enrollment_result = run_enrollment(source)

                                cam = CameraStream(stream_url=source)
                                if not cam.connect():
                                    print("❌ Could not reconnect Camera 1 after enrollment. Stopping.")
                                    ENTRANCE_RUNNING = False
                                    break

                                if enrollment_result == "success":
                                    new_customer = get_most_recent_customer()
                                    if new_customer:
                                        pin = create_session(new_customer["staff_id"], new_customer["name"])
                                        print(f"\n✅ Welcome Customer {new_customer['staff_id']}! Session PIN: {pin}")
                                        if _ON_EVENT:
                                            _ON_EVENT({"is_new": True, "name": new_customer["name"],
                                                       "staff_id": new_customer["staff_id"], "pin": pin})
                                        cooldown_lines = [
                                            f"Welcome Customer {new_customer['staff_id']}!",
                                            f"Your checkout PIN: {pin}",
                                            "(valid for this shopping session only)",
                                        ]
                                        cooldown_color = COL_BLACK
                                    else:
                                        cooldown_lines = ["Enrollment complete!"]
                                        cooldown_color = COL_BLACK
                                elif enrollment_result == "flagged":
                                    print("\n⚠️ Enrollment flagged for admin review — no PIN issued.")
                                    cooldown_lines = [
                                        "Enrollment flagged for review.",
                                        "Please see store staff.",
                                    ]
                                    cooldown_color = COL_YELLOW
                                else:
                                    cooldown_lines = ["Enrollment cancelled."]
                                    cooldown_color = COL_YELLOW

                                cooldown_until = time.time() + COOLDOWN_SECONDS
                                stable_since = None
                        else:
                            recognition_failures += 1
                            if recognition_failures >= MAX_RECOGNITION_FAILURES:
                                print(f"\n❌ Recognition failed {MAX_RECOGNITION_FAILURES} times in a row.")
                                cooldown_lines = ["Recognition failed.", "Please step back and try again."]
                                cooldown_color = COL_YELLOW
                                cooldown_until = time.time() + RECOGNITION_FAIL_COOLDOWN
                                recognition_failures = 0
                            stable_since = None
                else:
                    stable_since = None
                    recognition_failures = 0
                    guide_col = COL_YELLOW
                    msg = hint
            else:
                stable_since = None
                recognition_failures = 0
                if len(faces) > 1:
                    msg = "One person at a time, please"

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), guide_col, 2)

            _draw_guide_box(frame, color=box_color)

            cv2.putText(frame, "ENTRANCE CAMERA - Face Recognition Active", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CYAN, 1)
            cv2.putText(frame, msg, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_YELLOW, 1)
            cv2.imshow("Camera 1 - Entrance", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("\n  ⏸️ Entrance stream stopped by user.")
                ENTRANCE_RUNNING = False
                break

    finally:
        if mp_model is not None:
            mp_model.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_entrance_engine()


def run_entrance(camera_index=0):
    """Alias wrapper for sis_server.py import compatibility."""
    start_entrance_engine(camera_index)