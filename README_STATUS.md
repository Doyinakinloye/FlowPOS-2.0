# FlowPOS 2.0 — File Status (as of 2026-09-01)

See HOW_TO_RUN.md for setup + run instructions. This file replaces
several earlier, now-stale versions — consolidated here in one place
after finding real drift between what was delivered in chat and what
was actually on disk.

## Current entry flow (entrance + enrollment)
- **No name/ID typing.** `enrollment.py` auto-generates a Customer ID
  (`SI-001`, `SI-002`, ...) via `generate_staff_id()`. That ID is the
  only identifier, everywhere.
- **Liveness runs only during enrollment**, not for returning
  customers. A randomized 3-step head-turn challenge
  (`enrollment/liveness.py`, solvePnP-based) gates the 5-photo
  capture; entrance recognition is a straight detect → position →
  match, no challenge.
- **Position guide replaced entirely.** The old pixel-width "move
  closer/step back" check is gone. Both `entrance.py` and
  `enrollment.py` now use a shared, centered viewfinder-style guide
  box (`_face_position_check` / `_draw_guide_box`, both defined once
  in `enrollment.py`, imported into `entrance.py`) — accepted at any
  distance as long as the face is centered in the box; the box turns
  green when centered, with directional prompts otherwise.
- **Duplicate detection is real**, not just UI. A new enrollment's
  averaged embedding is checked against existing customers
  (`duplicate_checker.check_duplicate_from_embedding`); a match parks
  the candidate (photo + embeddings) via `save_pending_duplicate()`
  instead of saving it, returned to the caller as `"flagged"`.
  Resolved via **Admin Panel > Review Flagged Enrollment** — view the
  photo (opens in the OS default viewer), approve (enrolls anyway),
  or delete (discards). Tested end-to-end against a real DB copy.
- **3-strikes retry, in three separate places, each shaped for what
  actually repeats there:**
  - Liveness: 3 failed challenge attempts → enrollment cancelled.
  - Enrollment's 5-photo capture: 3 consecutive failed embedding
    extractions → restart the 5-photo sequence from zero. Isolated
    (non-consecutive) failures don't cost progress — verified both
    directions with an executed test, not just read through.
  - Entrance recognition: 3 consecutive failed extractions → visible
    "Recognition failed — step back and try again" message + short
    cooldown, instead of silently resetting forever with zero
    indication anything was wrong.
- **On-screen PIN display.** Both a returning-customer match and a
  freshly-enrolled customer show "Welcome Customer {ID}! Your
  checkout PIN: ####" directly on the camera window for 8 seconds —
  not just printed to a terminal a kiosk customer can't see.
- **Camera 1 auto-starts** as part of `main.py`'s startup sequence
  (not gated behind a menu selection) — but still prompts once for
  webcam vs. phone IP, restored after an earlier version accidentally
  removed that choice entirely.
- **Camera 2 (checkout)** supports the same webcam/phone-IP choice,
  fixed after discovering it only ever tried physical device indices
  and could never reach a phone stream.

## Real bugs found by actually running the code (not just reading it)
- `mediapipe.solutions` missing at runtime — an unpinned install
  pulled a version that dropped it. Pinned to `mediapipe==0.10.21`
  (confirmed via a live upstream bug report to still have it; v1's
  own pin, `0.10.9`, has no Python 3.12 wheel at all).
- `opencv-python` + `opencv-contrib-python` installed side by side
  corrupts the shared `cv2` module both provide — pinned to
  `opencv-contrib-python==4.11.0.86` only.
- Camera-resource leak in `entrance.py`: the MediaPipe model used to
  be created *before* the `try/finally` that releases the camera, so
  a setup-time crash (like the mediapipe.solutions one above) left
  the camera handle open forever, and `main.py`'s retry loop would
  reopen it every second without ever releasing the old one. Fixed by
  moving camera-dependent setup inside `try/finally`.
- `DUPLICATE_THRESHOLD` "single source of truth" wasn't actually
  shared — a circular import between `enrollment.py` and
  `duplicate_checker.py` made the latter silently fall back to its
  own hardcoded default. Same number, so invisible until checked with
  `is` (identity), not `==`. Fixed with a lazy import.
- `COL_WHITE` was used in `entrance.py` but only ever defined in
  `enrollment.py` — would have crashed the entrance loop with
  `NameError` on the very first frame of the next real-camera run,
  before a single face was ever detected. Caught by actually
  executing the loop with a mocked camera, not by reading the code.

## ⚠️ Still unverified against real hardware
- **Liveness yaw sign convention** — whether turning your head left
  actually reads as "LEFT" on screen, or backwards. One line to flip
  in `classify_pose()` if so.
- **Position-guide mirroring** — whether "Move left/right" prompts
  match the direction you actually need to move, or read backwards
  (depends on whether your camera preview is mirrored).
- **All timing/threshold constants** — liveness hold time and
  timeouts, recognition/capture stability windows, checkout detection
  confidence — are reasonable starting values, not tuned against real
  footage.

## Confirmed working (imports, boot, and functional tests actually run)
`sis_server.py`, `face_db.py`, `embedding_loader.py`,
`duplicate_checker.py`, `admin_panel.py`, `index.html`, `main.py`,
`camera_stream.py`, `services/enrollment_manager.py`,
`services/shopping_session.py`, `services/inventory.py`,
`enrollment/enrollment.py`, `enrollment/entrance.py`,
`enrollment/liveness.py`, `checkout/billing.py`,
`checkout/excel_logger.py`, `checkout/checkout.py`,
`m4/price_catalog.py` (coke=500, fanta=500, sprite=500, water=300),
`inventory.db`, `models/best.pt`.

## Still open / explicitly deferred
- **Checkout automation** — still requires pressing 'C' to trigger
  the scan; removing that was paused to focus on entry/enrollment
  first.
- **Block/ban customer list** — distinct from outright delete,
  mentioned early on, not yet built.
- **Past-transactions log** — depends on a real stored receipt
  existing first (checkout is currently screenshot-as-proof only).
- `enrollment/enrollment.py`'s `DUPLICATE_THRESHOLD` is now real and
  shared, but nothing has tuned the 0.75 value against real faces.

## 🗑️ Deliberately excluded — looks legacy
- `database.py` / `smart_store.db` — separate, unused schema; real
  data lives in `inventory.db` via `face_db.py` instead.
