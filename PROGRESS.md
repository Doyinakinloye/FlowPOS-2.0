# FlowPOS 2.0 — Progress Log

This is a running record of what's been built, what broke, and how it
got fixed — kept so the reasoning behind decisions doesn't get lost,
and so a future session (or a crash/rollback) has something more
useful to recover from than just a diff.

## Where this started

FlowPOS 1.0 was a continuous, always-on two-camera shelf-tracking
system — conceptually the same approach Amazon's Just Walk Out uses,
and it hit the same category of problem: expensive to run reliably,
dual-camera contention, unreliable event detection. It was retired
rather than tuned further.

FlowPOS 2.0 replaces continuous tracking with a two-checkpoint model:
an entrance camera that recognizes-or-enrolls a customer and issues a
four-digit shopping PIN, and a checkout camera that scans items and
totals a purchase. The two checkpoints never need to agree on the
same moment, which is what made v1 fragile.

## Core architecture (current)

- **Entrance (Camera 1, `enrollment/entrance.py`)** — continuous
  loop: detect a face → check it's centered in a viewfinder-style
  guide box (works at any distance, not a narrow pixel-width range)
  → hold steady briefly → match against enrolled customers. A match
  issues a PIN, shown on-screen for 8 seconds. No match hands off to
  enrollment.
- **Enrollment (`enrollment/enrollment.py`)** — no name or ID typing;
  a Customer ID (`SI-001`, `SI-002`, ...) is auto-generated. A
  randomized 3-step head-turn liveness challenge
  (`enrollment/liveness.py`, solvePnP-based) runs once, before the
  5-photo capture — this is the *only* place liveness runs; returning
  customers are never challenged. A new enrollment's embedding is
  checked against existing customers; a close match gets parked for
  admin review instead of silently saved as a possible duplicate.
- **Checkout (Camera 2, `checkout/checkout.py`)** — PIN entry, single-
  frame YOLOv8 item scan (manual 'C' keypress still required — this
  is the next planned piece of work), itemized total, screenshot as
  proof of payment (no real payment gateway, no stored receipt yet).
- **Admin Panel (`admin_panel.py`)** — active sessions, enrolled
  customers, inventory (view/add stock/set exact/reprice), flagged-
  enrollment review (view a real photo of the flagged attempt, opens
  in the OS default viewer; approve or reject).
- **Web dashboard (`sis_server.py`)** — Flask mirror of the terminal
  flow for browser-based control.

## Real bugs found and fixed this session

All of these were found by actually running the code — against a
real camera, or against a mocked pipeline built specifically to
reproduce the failure — not by code review alone.

1. **`mediapipe.solutions` missing at runtime.** An unpinned install
   pulled a MediaPipe version that had dropped the `mp.solutions` API
   the whole face pipeline depends on. Confirmed against a live
   upstream bug report. Pinned to `mediapipe==0.10.21` — the newest
   version confirmed to still ship it, and the only one with both
   that API and a Python 3.12 wheel (v1's own pin, `0.10.9`, has
   neither the fix nor a 3.12 wheel).
2. **`opencv-python` + `opencv-contrib-python` installed together**
   corrupts the shared `cv2` module both provide. Pinned to
   `opencv-contrib-python==4.11.0.86` only.
3. **Camera-resource leak.** The MediaPipe model in `entrance.py`
   used to be created *before* the `try/finally` that releases the
   camera, so a setup-time crash left the camera handle open
   indefinitely. Fixed by moving camera-dependent setup inside the
   `try` block.
4. **`DUPLICATE_THRESHOLD` wasn't actually shared.** A circular
   import between `enrollment.py` and `duplicate_checker.py` made the
   latter silently fall back to its own hardcoded default — same
   value, so invisible until checked with `is` (identity) instead of
   `==`. Fixed with a lazy import inside the function that needs it.
5. **`COL_WHITE` used but never defined in `entrance.py`** — would
   crash the entrance loop with `NameError` on the very first frame,
   before a single face was ever detected. Caught by actually
   executing the loop with a mocked camera, not by reading the code.
6. **The real crash-loop bug**: `_mediapipe_face_crop(...) or
   upscaled` used Python's `or` to fall back when the secondary
   detection returned `None`. That only works because `None` is
   falsy — the moment detection actually *succeeds* and returns a
   real multi-element numpy array, Python can't evaluate that array's
   truthiness (`ValueError: truth value of an array... is
   ambiguous`). This is exactly why the entrance camera flashed on
   and immediately died in a loop on the first real test — it only
   crashed on a *successful* detection. Fixed with an explicit
   `is None` check in both `enrollment.py` and `entrance.py`.
7. **`protobuf` version mismatch** after an interrupted `tf-keras`
   install (which itself ran out of disk space trying to force a
   TensorFlow upgrade that wasn't necessary). Fixed by reinstalling
   the matching `protobuf==4.25.9`, and by installing a
   version-matched `tf-keras==2.19.0` instead of the latest (which
   would have pulled `tensorflow>=2.21` and re-triggered the same
   problem).
8. **'Q' and "Stop Entrance Engine" silently did nothing** — only
   Ctrl+C (killing the whole process) actually stopped the camera.
   Root cause: pressing 'Q' only `break`s entrance.py's own loop, it
   never set entrance.py's `ENTRANCE_RUNNING` flag — so main.py's
   outer daemon loop had no way to tell an intentional stop from
   "try again" and just kept reconnecting. Separately, main.py's
   `stop_entrance()` only touched its *own* disconnected flag, never
   entrance.py's — so a genuinely running camera loop never heard
   about it at all. Fixed by having 'Q' explicitly set the flag, and
   having main.py call `stop_entrance_engine()` directly. Confirmed
   against real hardware after the fix — the menu correctly showed
   "Entrance Status: STOPPED" instead of staying stuck on ACTIVE.

## Verified on real hardware (not just imports/mocks)

- Full enrollment: auto-generated ID, liveness passed on the first
  attempt, all 5 photos captured, saved to the database.
- The same person recognized immediately afterward, with **no**
  liveness re-trigger, PIN issued and shown on-screen.
- Admin panel: view customers, check inventory, add stock, delete
  customer — all exercised with no crashes.
- Stopping the entrance engine (both 'Q' and the menu option) now
  genuinely releases the camera instead of silently continuing.

## Known, not yet fixed

- `_detect_faces_mp()` in `enrollment.py` creates a brand-new
  MediaPipe model on *every frame* of the 5-photo capture loop
  instead of reusing one — wasteful, and the likely source of visible
  per-frame lag during capture (confirmed via a real demo recording:
  dozens of "Feedback manager..." warnings firing in rapid
  succession, one per frame). Not yet fixed.
- Liveness yaw-sign convention (does "turn left" actually read as
  LEFT on screen, or backwards) and the position-guide box's
  left/right mirroring are both still unconfirmed against what the
  camera preview actually showed — the demo recording only captured
  the VS Code window, not the separate OS camera window.
- All timing/threshold constants (liveness hold time, stability
  windows, detection confidence) are reasonable starting values, not
  tuned against real footage.

## Deliberately deferred

- **Checkout automation** — still requires pressing 'C' to trigger
  the scan.
- **Block/ban customer list**, distinct from outright delete.
- **Past-transactions log** and a real stored receipt (checkout is
  currently screenshot-as-proof only; `billing.py` and
  `excel_logger.py` are dead code from v1, called by nothing in the
  current checkout flow).
- **Inventory–checkout link** — a completed sale does not currently
  decrement recorded stock anywhere in the code.
- No admin login/authentication gate on the admin panel.

## Files changed from the original upload

`requirements.txt`, `main.py`, `admin_panel.py`,
`services/enrollment_manager.py`, `services/inventory.py` (new),
`enrollment/enrollment.py`, `enrollment/entrance.py`,
`enrollment/liveness.py` (new), `checkout/checkout.py`,
`m4/price_catalog.py`. Everything else (`face_db.py`,
`embedding_loader.py`, `camera_stream.py`, `sis_server.py`,
`services/shopping_session.py`, `checkout/billing.py`,
`checkout/excel_logger.py`, `duplicate_checker.py`) is untouched from
what was originally uploaded.
