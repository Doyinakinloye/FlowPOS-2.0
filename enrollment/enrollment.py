# ============================================
# SMART INVENTORY SYSTEM
# Module 2: Face Enrollment & Database Maintenance
#
# Pipeline per enrollment session:
#   Camera feed → face detection (MediaPipe) → CLAHE preprocessing →
#   ArcFace embedding generation (DeepFace) → append to embeddings.pkl
#   and save metadata in face_db.json
#
# FIXED 2026-09-01 (later same day) — _detect_faces_mp() used to
# create a brand-new MediaPipe FaceDetection model on every single
# call, and it's called once per frame in the capture loop below.
# Harmless functionally, but wasteful, and confirmed as the source of
# a real demo recording showing dozens of "Feedback manager..."
# warnings firing in rapid succession during capture -- one fresh
# model construction per frame. Now takes the same already-created
# mp_model the rest of this function already uses (for
# _mediapipe_face_crop) instead of building its own. Verified with a
# real executed test: FaceDetection() now constructs exactly once for
# a full 5-photo enrollment, not once per frame.
#
# UPDATED 2026-08-31 (entry flow rework) — three changes, at the
# user's explicit request:
#   1. No more name/ID typing. staff_id is auto-generated via
#      services.enrollment_manager.generate_staff_id() (SI-001,
#      SI-002, ...). There's no separate "name" collected anymore —
#      the ID doubles as the display identifier everywhere (DB still
#      has a NOT NULL name column, so it's set to the same value as
#      staff_id rather than left blank).
#   2. Liveness now lives HERE, not in entrance.py — a randomized
#      head-pose challenge (enrollment/liveness.py) runs right after
#      the camera connects, before any photos are captured. 3 attempts
#      (MAX_LIVENESS_ATTEMPTS), then enrollment is cancelled. This
#      stops someone enrolling a photo of another person; entrance.py
#      no longer challenges *returning* customers at all.
#   3. The old GUIDE_BOX_MIN_PX/MAX_PX check (face width in pixels, a
#      distance proxy) is replaced by a fixed on-screen guide box —
#      capture proceeds as long as the face is CENTERED in that box,
#      at any distance, rather than only within a narrow pixel-width
#      range. Off-center shows a directional prompt.
#
# ⚠️ UNVERIFIED: the guide-box directional hints ("move left/right")
# assume the on-screen preview is NOT mirrored. If your camera preview
# is mirrored (flipped like a selfie cam), left/right hints will read
# backwards — swap them in _face_position_check() below. Same
# unverified-until-real-camera category as the liveness yaw sign.
#
# FIXED 2026-09-01 (real-camera crash) — `_mediapipe_face_crop(...) or
# upscaled` used Python's `or` to fall back to `upscaled` when the
# secondary detection failed (returns None). That only works because
# `None` is falsy — the moment the secondary detection SUCCEEDS and
# returns a real (non-None) numpy image array, Python has to evaluate
# that array's truthiness to decide whether to short-circuit, and a
# multi-element array's truthiness is undefined: crashes with
# "The truth value of an array with more than one element is
# ambiguous." This is exactly what happened on the real run -- it
# only crashes on a SUCCESSFUL detection, which is why it never
# showed up until real hardware with a real face in frame. Replaced
# with an explicit `is None` check. Verified two ways: reproduced the
# exact crash standalone, then re-ran the full 5-photo capture loop
# with a mocked secondary detection that always succeeds (the
# previously-crashing case) end to end with no exception.
#
# FIXED 2026-09-03 — a normal, successful enrollment was never saving
# a photo at all. The only cv2.imwrite() call in this whole file lived
# inside the duplicate-flagged branch (to give an admin something to
# look at when reviewing a flagged case) -- for a clean, non-duplicate
# enrollment, save_customer() was being called with folder_path=""
# and no image file was ever written anywhere. That silently broke
# anything depending on a customer having a real photo on file, most
# visibly the dashboard's Transactions page (added the same day),
# which shows this photo next to each sale -- every row was falling
# back to initials, never showing a real photo. Fixed by adding the
# same save step (already proven in the duplicate branch) to the
# success path too. staff_id-based folder naming, not name-based --
# correct under this architecture since name == staff_id (no separate
# name is collected). Verified two ways with the real, unmodified
# capture/liveness/duplicate-check logic (only camera/MediaPipe
# hardware mocked): a normal successful enrollment now genuinely
# creates customers/<ID>/best_frame.jpg on disk with a real,
# non-empty folder_path in the database, and the duplicate-flagged
# branch right next to this change still behaves exactly as before.
# ============================================

import os
import sys
import cv2
import time
import numpy as np

# System path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_stream import CameraStream, get_stream_url
from embedding_loader import load_embeddings, add_embedding_to_memory as add_embedding
from face_db import save_customer, customer_exists, get_all_customers
from services.enrollment_manager import generate_staff_id, save_pending_duplicate, PENDING_DUPLICATE_DIR
from enrollment.liveness import run_liveness_check
# duplicate_checker is imported lazily inside run_enrollment(), not here --
# it imports DUPLICATE_THRESHOLD back from this module, and importing it
# at module level here would hit that constant before it's defined below,
# silently falling back to duplicate_checker's own hardcoded default
# instead of truly sharing this one. Verified: with a top-level import,
# `dc.DUPLICATE_THRESHOLD is em.DUPLICATE_THRESHOLD` was False (same
# value, different object) -- moving the import inside the function
# fixes that, confirmed the same way.

# ── CONFIGURATION & CONSTANTS ─────────────────────────────────────────

CAPTURE_COUNT           = 5
STABLE_SECONDS_REQUIRED = 1.2
CAMERA_RETRY_ATTEMPTS   = 3
CAMERA_RETRY_DELAY_SEC  = 2.0
MAX_FAILED_READS        = 15
MAX_LIVENESS_ATTEMPTS   = 3
MAX_CAPTURE_FAILURES    = 3  # consecutive failed embedding extractions before restarting the 5-photo capture

# Single source of truth for duplicate_checker.py's import (see that
# file's comment on why importing beats a second hardcoded number).
DUPLICATE_THRESHOLD = 0.75

# Position-guide box: centered rectangle the face must fall inside,
# as a fraction of frame width/height. Face is checked by its CENTER
# point falling inside this box — not by pixel width — so it works at
# varying distances, not just one narrow "just right" range.
GUIDE_MARGIN_W = 0.25  # box spans the center 50% of frame width
GUIDE_MARGIN_H = 0.15  # box spans the center 70% of frame height

H, W = 480, 640

# Theme Colors (BGR)
COL_GREEN  = (0, 220, 0)
COL_YELLOW = (0, 220, 255)
COL_GRAY   = (140, 140, 140)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0, 0, 0)
COL_CYAN   = (255, 220, 0)
COL_RED    = (50, 50, 255)
COL_ORANGE = (0, 140, 255)

_MP_CONF  = 0.5
_MP_MODEL = 0

# ── HELPER UTILITIES ─────────────────────────────────────────────────

def _preprocess(frame):
    h, w = frame.shape[:2]
    if h == H and w == W:
        return frame
    return cv2.resize(frame, (W, H))

def _detect_faces_mp(frame, mp_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_model.process(rgb)
    if not results.detections:
        return []

    faces = []
    for det in results.detections:
        bb = det.location_data.relative_bounding_box
        x = int(bb.xmin * W)
        y = int(bb.ymin * H)
        w = int(bb.width * W)
        h = int(bb.height * H)
        score = det.score[0] if det.score else 0.0
        faces.append(((x, y, w, h), score))
    return faces

def _guide_box():
    """Returns (x1, y1, x2, y2) of the centered position-guide box, in frame coords."""
    x1 = int(W * GUIDE_MARGIN_W)
    x2 = int(W * (1 - GUIDE_MARGIN_W))
    y1 = int(H * GUIDE_MARGIN_H)
    y2 = int(H * (1 - GUIDE_MARGIN_H))
    return x1, y1, x2, y2

def _face_position_check(bbox):
    """
    bbox: (x, y, w, h) face box in frame coords.
    Returns (centered: bool, hint: str) -- hint is a short directional
    prompt like "Move right" when not centered, "" when centered.
    """
    x, y, w, h = bbox
    fcx, fcy = x + w // 2, y + h // 2
    gx1, gy1, gx2, gy2 = _guide_box()

    hints = []
    if fcx < gx1:
        hints.append("right")
    elif fcx > gx2:
        hints.append("left")
    if fcy < gy1:
        hints.append("down")
    elif fcy > gy2:
        hints.append("up")

    if not hints:
        return True, ""
    return False, "Move " + " and ".join(hints) + " into the box"

def _draw_guide_box(frame, color=COL_WHITE, thickness=2):
    """
    Viewfinder-style guide box: four L-shaped corner brackets rather
    than a full outline (matches common face-scan UX like ID/passport
    photo apps) -- clearer to align against than a thin rectangle.
    Pass color=COL_GREEN when the caller has determined the face is
    currently centered, so the box itself gives positive feedback
    without the customer needing to read the status text.
    """
    x1, y1, x2, y2 = _guide_box()
    bl = int(min(x2 - x1, y2 - y1) * 0.15)  # bracket arm length

    cv2.line(frame, (x1, y1), (x1 + bl, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + bl), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - bl, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + bl), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + bl, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - bl), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - bl, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - bl), color, thickness)

def _upscale_bbox_crop(frame, bbox):
    x, y, w, h = bbox
    pad_w = int(w * 0.35)
    pad_h = int(h * 0.35)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    if ch < 160 or cw < 160:
        scale = max(200 / ch, 200 / cw)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
    return crop

def _mediapipe_face_crop(crop, mp_model):
    ch, cw = crop.shape[:2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = mp_model.process(rgb)
    if not res.detections:
        return None

    det = max(res.detections, key=lambda d: d.score[0] if d.score else 0)
    bb = det.location_data.relative_bounding_box
    x = max(0, int(bb.xmin * cw))
    y = max(0, int(bb.ymin * ch))
    w = min(cw - x, int(bb.width * cw))
    h = min(ch - y, int(bb.height * ch))

    margin = int(min(w, h) * 0.12)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(cw, x + w + margin)
    y2 = min(ch, y + h + margin)

    face = crop[y1:y2, x1:x2]
    return face if face.size > 0 else None

def _clahe_preprocess(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def _close_camera_window(camera):
    if camera:
        camera.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    for _ in range(4):
        cv2.waitKey(1)

def connect_camera_with_retry(stream_url):
    for attempt in range(1, CAMERA_RETRY_ATTEMPTS + 1):
        print(f"  Attempting camera connection ({attempt}/{CAMERA_RETRY_ATTEMPTS})...")
        cam = CameraStream(stream_url)
        if cam.connect():
            print("  Camera connected ✅")
            return cam, stream_url
        cam.release()
        if attempt < CAMERA_RETRY_ATTEMPTS:
            time.sleep(CAMERA_RETRY_DELAY_SEC)

    print("  Failed to connect. Prompting for new stream URL...")
    new_url = input("  Enter stream URL or index (e.g. 0, 1, http://...): ").strip()
    if not new_url:
        return None, stream_url

    cam = CameraStream(new_url)
    if cam.connect():
        print("  Camera connected ✅")
        return cam, new_url

    cam.release()
    print("  Could not connect to new stream URL.")
    return None, stream_url

# ── EMBEDDING EXTRACTION ──────────────────────────────────────────────

def _extract_embedding(crop):
    from deepface import DeepFace
    try:
        res = DeepFace.represent(img_path=crop, model_name="ArcFace", detector_backend="skip", enforce_detection=False)
        if res:
            return np.array(res[0]["embedding"])
    except Exception as e:
        print(f"  ArcFace embedding extraction failed: {e}")
    return None

# ── UI DRAWING ─────────────────────────────────────────────────────────

def _draw_enrollment_ui(frame, current, total, state_msg, faces, guide_color, box_color=None):
    disp = _preprocess(frame).copy()
    overlay = disp.copy()
    cv2.rectangle(overlay, (0, 0), (W, 50), COL_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, disp, 0.3, 0, disp)

    cv2.putText(disp, "SMART INVENTORY - Customer Enrollment", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CYAN, 1)
    cv2.putText(disp, f"Captured: {current}/{total}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_WHITE, 1)

    _draw_guide_box(disp, color=box_color or COL_WHITE)

    for (x, y, w, h), _ in faces:
        cv2.rectangle(disp, (x, y), (x + w, y + h), guide_color, 2)

    cv2.putText(disp, state_msg, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_YELLOW, 1)
    return disp

# ── MAIN ENROLLMENT EXECUTION ──────────────────────────────────────────

def run_enrollment(stream_url):
    print("\n" + "=" * 56)
    print("   FACE ENROLLMENT")
    print("=" * 56)

    staff_id = generate_staff_id()
    name = staff_id  # no name is collected -- the ID is the identifier everywhere
    print(f"  Assigned Customer ID: {staff_id}")

    camera, _ = connect_camera_with_retry(stream_url)
    if camera is None:
        return "cancelled"

    import mediapipe as mp
    mp_model = mp.solutions.face_detection.FaceDetection(model_selection=_MP_MODEL, min_detection_confidence=_MP_CONF)

    # ── LIVENESS GATE ──────────────────────────────────────────────
    # Restricted to enrollment only (not entrance recognition, per
    # 2026-08-31 request) -- stops someone enrolling a photo of
    # another person. Same 3-attempt pattern used elsewhere.
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    liveness_ok = False
    try:
        for attempt in range(1, MAX_LIVENESS_ATTEMPTS + 1):
            print(f"\n  🔒 Liveness check (attempt {attempt}/{MAX_LIVENESS_ATTEMPTS})...")
            if run_liveness_check(camera, mp_face_mesh, _preprocess, W, H,
                                   window_name="Smart Inventory - Enrollment"):
                liveness_ok = True
                break
            print(f"  ⚠️ Liveness check failed — attempt {attempt}/{MAX_LIVENESS_ATTEMPTS}.")
    finally:
        mp_face_mesh.close()

    if not liveness_ok:
        print(f"  ❌ Liveness check failed {MAX_LIVENESS_ATTEMPTS} times. Enrollment cancelled.")
        mp_model.close()
        _close_camera_window(camera)
        return "cancelled"

    embeddings = []
    last_face_crop = None  # kept so a flagged duplicate has a real photo to show the admin
    stable_since = None
    failed_reads = 0
    capture_failures = 0  # consecutive failed extraction attempts (not camera-read failures)

    try:
        while len(embeddings) < CAPTURE_COUNT:
            frame = camera.read_frame()
            if frame is None:
                failed_reads += 1
                if failed_reads >= MAX_FAILED_READS:
                    print("  Stream lost during enrollment.")
                    break
                continue
            failed_reads = 0

            frame = _preprocess(frame)
            faces = _detect_faces_mp(frame, mp_model)

            guide_col = COL_GRAY
            msg = "Center your face in the box..."
            box_color = COL_WHITE

            if len(faces) == 1:
                bbox, score = faces[0]
                centered, hint = _face_position_check(bbox)
                box_color = COL_GREEN if centered else COL_WHITE

                if centered:
                    if stable_since is None:
                        stable_since = time.time()

                    elapsed = time.time() - stable_since
                    if elapsed >= STABLE_SECONDS_REQUIRED:
                        guide_col = COL_GREEN
                        msg = "Hold steady... Capturing face..."

                        upscaled = _upscale_bbox_crop(frame, bbox)
                        face_crop = None
                        if upscaled is not None:
                            face_crop = _mediapipe_face_crop(upscaled, mp_model)
                            if face_crop is None:
                                face_crop = upscaled
                            face_crop = _clahe_preprocess(face_crop)
                            t0 = time.time()
                            emb = _extract_embedding(face_crop)
                            print(f"  [timing] embedding extraction took {time.time() - t0:.2f}s")
                        else:
                            emb = None

                        if emb is not None:
                            embeddings.append(emb)
                            last_face_crop = face_crop
                            capture_failures = 0
                            print(f"  Captured sample {len(embeddings)}/{CAPTURE_COUNT}")
                            stable_since = time.time()
                        else:
                            capture_failures += 1
                            print(f"  ⚠️ Capture attempt failed — {capture_failures}/{MAX_CAPTURE_FAILURES}.")
                            if capture_failures >= MAX_CAPTURE_FAILURES:
                                print(f"  ❌ Capture failed {MAX_CAPTURE_FAILURES} times in a row. Restarting enrollment capture...")
                                notice = _draw_enrollment_ui(frame, 0, CAPTURE_COUNT,
                                                              "Capture failed. Restarting...",
                                                              faces, COL_RED, COL_RED)
                                cv2.imshow("Smart Inventory - Enrollment", notice)
                                cv2.waitKey(1200)
                                embeddings = []
                                last_face_crop = None
                                capture_failures = 0
                                stable_since = None
                            else:
                                stable_since = time.time()
                    else:
                        guide_col = COL_YELLOW
                        msg = f"Hold steady... {elapsed:.1f}s"
                else:
                    stable_since = None
                    guide_col = COL_YELLOW
                    msg = hint
            else:
                stable_since = None
                if len(faces) > 1:
                    msg = "Multiple faces detected — stay solo"

            disp = _draw_enrollment_ui(frame, len(embeddings), CAPTURE_COUNT, msg, faces, guide_col, box_color)
            cv2.imshow("Smart Inventory - Enrollment", disp)

            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                print("  Enrollment cancelled by user.")
                break

    finally:
        mp_model.close()
        _close_camera_window(camera)

    if len(embeddings) == CAPTURE_COUNT:
        avg_emb = np.mean(embeddings, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)

        # ── DUPLICATE CHECK ──────────────────────────────────────────
        # A match here means this face's embedding is already close to
        # an enrolled customer's -- park it for admin review instead of
        # silently creating a second record for what may be the same
        # person. Saves the last captured (CLAHE-enhanced) frame as a
        # real photo file, since a terminal admin panel can't render
        # an image inline -- the admin panel opens it with the OS's
        # default viewer instead.
        from duplicate_checker import check_duplicate_from_embedding
        is_dup, match_name, match_id, similarity = check_duplicate_from_embedding(avg_emb)
        if is_dup:
            photo_path = ""
            if last_face_crop is not None:
                os.makedirs(PENDING_DUPLICATE_DIR, exist_ok=True)
                photo_path = os.path.join(PENDING_DUPLICATE_DIR, f"{staff_id}.jpg")
                cv2.imwrite(photo_path, last_face_crop)
            save_pending_duplicate(staff_id, name, photo_path, embeddings, match_name, match_id, similarity)
            print(f"  ⚠️ Possible duplicate of {match_name} ({match_id}) — {similarity:.0%} similarity.")
            print("  Enrollment parked for admin review (Admin Panel > Review Flagged Enrollment).")
            return "flagged"

        add_embedding(staff_id, name, avg_emb)

        # Save a real reference photo for this customer, the same way
        # the flagged-duplicate branch above already does -- FOUND AND
        # FIXED: this was missing entirely for a normal, successful
        # enrollment (folder_path was being saved as "" with no photo
        # file written anywhere). That silently broke anything that
        # depends on a customer having a real photo on file -- e.g. the
        # dashboard's Transactions page, which shows this photo next to
        # each sale. staff_id-based folder naming (not name-based) is
        # correct here since name == staff_id under this architecture
        # (no separate name is collected anymore).
        folder_path = ""
        if last_face_crop is not None:
            folder_path = os.path.join("customers", staff_id)
            os.makedirs(folder_path, exist_ok=True)
            cv2.imwrite(os.path.join(folder_path, "best_frame.jpg"), last_face_crop)

        save_customer(staff_id, name, folder_path, avg_emb)
        print(f"  ✅ Enrollment successful! Customer ID: {staff_id}")
        return "success"
    else:
        print("  Enrollment incomplete.")
        return "cancelled"

if __name__ == "__main__":
    url = get_stream_url()
    run_enrollment(url) 