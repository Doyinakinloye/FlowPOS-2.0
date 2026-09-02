# ============================================
# SMART INVENTORY SYSTEM
# Module: Checkout Flow
# Description: Automatic Pause/Resume of Entrance Engine during Checkout
#
# UPDATED 2026-08-31 — three changes from the version you had:
#   1. scan_items_horizontal() no longer returns a hardcoded fake list.
#      Pressing 'C' ran the real best.pt model on that one frame.
#   2. Currency display switched from "$" to "N" (Naira).
#   3. The "Camera 2 hardware not detected" fallback no longer invents
#      a fake purchase — returns an empty list instead, which
#      run_checkout()'s "no items scanned" check already handles safely.
#
# REWORKED 2026-09-01 — full redesign, at the user's explicit request:
#   1. Arrangement instructions are shown BEFORE PIN entry, not after —
#      the customer knows what to do while they're still setting up.
#   2. PIN entry now directly triggers the camera — no separate "press
#      Enter to open Camera 2" step. One action instead of two.
#   3. Camera 2's source (webcam vs. phone IP) is asked once per
#      program run and cached (_CHECKOUT_STREAM_SOURCE), not re-asked
#      on every checkout — matches how entrance.py's camera source is
#      remembered, and makes "PIN triggers camera" actually feel
#      automatic rather than gated behind a repeated prompt.
#   4. Scanning is now continuous instead of manual-'C'-required: runs
#      a detection pass every ~0.6s (not every frame — YOLO inference
#      is heavier than the face detection used elsewhere, and this is
#      CPU-only) and auto-confirms once the same item set is seen 3
#      times in a row — same "3 consistent reads" pattern already used
#      for liveness/capture/recognition elsewhere in this app. 'C'
#      still works as a manual override to force an immediate read.
#   5. If nothing stabilizes within SCAN_TIMEOUT_SEC, the customer gets
#      ONE more attempt rather than being dropped straight back to
#      entrance; only a second timeout ends the session. This is a
#      smaller number of attempts than the 3 used elsewhere in the app
#      — a deliberate choice for this specific flow, not copied from
#      the other retry patterns.
#   6. After a stable read, a summary screen (items + total) shows
#      before anything is finalized — 'A' to accept, 'R' to rescan
#      (unlimited, unlike the timeout-retry above — a customer
#      reviewing a wrong result should always be able to just try
#      again), 'Q' to cancel. Item-detection accuracy is unverified
#      against real footage, so a review step before charging anyone
#      matters, not just a nice-to-have.
#   7. There is no customer-facing "print receipt" step anymore.
#      Accepting saves the sale via face_db.py's existing
#      save_transaction() -- already used by the System Info screen's
#      "Total sales" figure, so reused rather than building a second,
#      competing transactions table. The admin panel's new "View Past
#      Transactions" reads it back, instead of a screenshot or a
#      printer this project has no way to actually drive.
#
# FIXED 2026-09-02 -- a completed sale never touched recorded stock
# anywhere in the code (documented as a known gap, now closed). Each
# purchased item now calls add_stock(name, -qty) right after the
# transaction saves successfully -- that function already existed
# with exactly this in mind, just was never actually called from
# here. Not clamped at zero on purpose (see the comment at the call
# site in run_checkout()).
#
# FIXED 2026-09-02 (later same day) -- Admin Panel > View All Enrolled
# Customers always showed Purchases: 0 / Spent: N0.00 even after real
# transactions -- same root cause as the inventory bug just above.
# face_db.py already had update_customer_stats(staff_id, amount)
# ("Called after each transaction" per its own docstring), it just
# was never actually called from here. Added right next to
# save_transaction(), same place add_stock() lives.
#
# NEW 2026-09-02 (web UI groundwork) -- added an optional event hook,
# same shape as entrance.py's: fires once per completed sale with
# {staff_id, name, items, total}, does nothing unless sis_server.py
# registers a listener. This is what will let the live dashboard show
# "customer just checked out" in real time later -- zero effect on
# the terminal flow on its own.
#
# Not yet tuned against real footage: CONF_THRESHOLD, the 3-reads
# stability count, the 0.6s poll interval, and SCAN_TIMEOUT_SEC are
# reasonable starting values, not calibrated against a real tray under
# real lighting.
# ============================================

import os
import sys
import cv2
import time
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.shopping_session import (
    get_session_by_pin,
    close_session,
    pause_entrance,
    resume_entrance
)
from services.inventory import add_stock
from face_db import save_transaction, get_all_transactions, update_customer_stats
from camera_stream import get_stream_url
from m4.price_catalog import get_price

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "best.pt"
)
CONF_THRESHOLD = 0.5  # detection confidence cutoff — tune against real footage

STABILITY_CHECKS_REQUIRED = 3   # consecutive matching detection passes before auto-confirming
DETECTION_INTERVAL_SEC    = 0.6 # how often to run a detection pass while waiting
SCAN_TIMEOUT_SEC          = 60  # how long to wait for a stable read before offering a retry
MAX_SCAN_ATTEMPTS         = 2   # this scan specifically gets 2 tries total, not the 3 used elsewhere

_MODEL = None
_CHECKOUT_STREAM_SOURCE = None  # cached camera 2 source -- asked once per program run, then reused

# Optional hook, set via set_event_hook(). Fires exactly once per
# completed sale, with {"staff_id": str, "name": str, "items": list,
# "total": number}. Does nothing by default -- purely additive, zero
# effect on the terminal flow unless something (sis_server.py)
# explicitly registers a hook. Same pattern as entrance.py's hook, for
# the same reason: a plain True/False return doesn't carry enough for
# a dashboard log entry (no items, no total, no customer name).
_ON_EVENT = None


def set_event_hook(fn):
    """Register fn(event_dict) to be called on every completed sale.
    Pass None to clear it."""
    global _ON_EVENT
    _ON_EVENT = fn


def _get_model():
    """Lazy-load YOLO so importing this module doesn't require ultralytics
    unless checkout is actually run."""
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Item-detection model not found at {MODEL_PATH!r}. "
                f"Update MODEL_PATH at the top of checkout.py if best.pt "
                f"lives somewhere else in your project."
            )
        _MODEL = YOLO(MODEL_PATH)
    return _MODEL


def _detect_items(frame):
    """
    Runs one detection pass on `frame`. Returns (items, annotated_frame)
    where items is [{"name": label, "price": <N>, "qty": <count>}, ...]
    """
    model = _get_model()
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return [], frame

    boxes = results[0].boxes
    labels = [model.names[int(c)] for c in boxes.cls.cpu().numpy()]
    counts = Counter(labels)

    items = [
        {"name": label, "price": get_price(label), "qty": qty}
        for label, qty in counts.items()
    ]
    annotated = results[0].plot()
    return items, annotated


def _resolve_source(camera_index):
    """None/empty -> webcam 0 (matches get_stream_url()'s documented
    contract: 'Returns the full stream URL, or None for webcam').
    Digit strings -> int. Anything else (a URL, e.g. an IP Webcam
    stream) passes through unchanged."""
    if camera_index is None or (isinstance(camera_index, str) and camera_index.strip() == ""):
        return 0
    if isinstance(camera_index, str) and camera_index.strip().lstrip("-").isdigit():
        return int(camera_index.strip())
    return camera_index


def _get_checkout_camera_source():
    """Asks for Camera 2's source once per program run, then reuses it
    for every subsequent checkout -- so PIN entry actually feels like it
    triggers the camera directly, not gated behind a repeated prompt."""
    global _CHECKOUT_STREAM_SOURCE
    if _CHECKOUT_STREAM_SOURCE is None:
        print("\n--- Camera 2: CHECKOUT (item scanning) ---")
        _CHECKOUT_STREAM_SOURCE = get_stream_url()
    return _CHECKOUT_STREAM_SOURCE


def _draw_scan_overlay(frame, msg_lines, guide_color=(0, 255, 255)):
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (50, int(H * 0.4)), (W - 50, int(H * 0.85)), guide_color, 2)
    cv2.putText(frame, "LAY ITEMS HORIZONTALLY IN THIS AREA", (60, int(H * 0.38)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, guide_color, 2)
    y = H - 15 - (len(msg_lines) - 1) * 22
    for line in msg_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        y += 22


def _run_one_scan_attempt(cap):
    """
    Continuously polls the camera, running a detection pass every
    DETECTION_INTERVAL_SEC. Auto-confirms once the same item set is
    read STABILITY_CHECKS_REQUIRED times in a row. 'C' force-confirms
    whatever's currently detected; 'Q' cancels the whole checkout.

    Returns:
      ("stable", items)  -- got a confirmed reading
      ("timeout", None)  -- SCAN_TIMEOUT_SEC passed with no stable reading
      ("cancelled", None) -- user pressed 'Q'
    """
    start_time = time.time()
    last_signature = None
    stable_count = 0
    last_items = []
    last_detect_time = 0.0

    while True:
        elapsed = time.time() - start_time
        if elapsed > SCAN_TIMEOUT_SEC:
            return "timeout", None

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        now = time.time()
        if now - last_detect_time >= DETECTION_INTERVAL_SEC:
            last_detect_time = now
            try:
                t0 = time.time()
                items, annotated = _detect_items(frame)
                print(f"  [timing] YOLO detection pass took {time.time() - t0:.2f}s")
            except Exception as e:
                print(f"  Detection error: {e}")
                items, annotated = [], frame

            signature = tuple(sorted((it["name"], it["qty"]) for it in items))
            if items and signature == last_signature:
                stable_count += 1
            else:
                stable_count = 0
            last_signature = signature
            last_items = items

            if items and stable_count >= STABILITY_CHECKS_REQUIRED:
                cv2.imshow("Camera 2 - Item Scanner", annotated)
                cv2.waitKey(1200)
                return "stable", items

        remaining = max(0, int(SCAN_TIMEOUT_SEC - elapsed))
        if last_items:
            detected_str = ", ".join(f"{it['qty']}x {it['name']}" for it in last_items)
            msg_lines = [
                f"Detected: {detected_str}",
                f"Stabilizing... {stable_count}/{STABILITY_CHECKS_REQUIRED}",
                f"'C' confirm now | 'Q' cancel | {remaining}s left",
            ]
        else:
            msg_lines = [
                "Looking for items...",
                f"'C' confirm now | 'Q' cancel | {remaining}s left",
            ]
        _draw_scan_overlay(frame, msg_lines)
        cv2.imshow("Camera 2 - Item Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('c'), ord('C')):
            if last_items:
                return "stable", last_items
            # nothing detected yet -- 'C' has nothing to confirm, keep waiting
        elif key in (ord('q'), ord('Q')):
            return "cancelled", None


def _show_summary_and_get_decision(cap, items):
    """
    Shows the itemized total on the same camera window and waits for
    'A' (accept), 'R' (rescan -- unlimited, unlike the timeout-retry
    in scan_items_horizontal), or 'Q' (cancel).
    """
    total = sum(it["price"] * it["qty"] for it in items)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        y = 40
        cv2.putText(frame, "ITEM SUMMARY", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 40
        for it in items:
            line = f"{it['qty']}x {it['name']} - N{it['price'] * it['qty']}"
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 28
        y += 12
        cv2.putText(frame, f"TOTAL: N{total}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
        y += 50
        cv2.putText(frame, "Press 'A' to Accept and Pay", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1)
        y += 28
        cv2.putText(frame, "Press 'R' to Rescan", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        y += 28
        cv2.putText(frame, "Press 'Q' to Cancel", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 255), 1)

        cv2.imshow("Camera 2 - Item Scanner", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('a'), ord('A')):
            return "accept"
        elif key in (ord('r'), ord('R')):
            return "rescan"
        elif key in (ord('q'), ord('Q')):
            return "cancelled"


def scan_items_horizontal(camera_index=1):
    """
    Owns Camera 2 for the whole scan-review cycle: keeps detecting
    automatically, offers one retry if nothing stabilizes in time,
    and shows an accept/rescan review screen once something does.
    Returns the accepted item list, or [] if cancelled/gave up.
    """
    print("\n[Camera 2] Opening Item Scanning Camera...")
    source = _resolve_source(camera_index)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened() and isinstance(source, int):
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"  Camera 2 hardware not detected (source: {source!r}). Checkout cannot proceed without a real scan.")
        return []

    print("\n" + "=" * 56)
    print(" ITEM SCANNING ACTIVE")
    print(" Items are checked automatically -- no key needed unless you want to.")
    print("=" * 56)

    accepted_items = []
    attempts_left = MAX_SCAN_ATTEMPTS

    try:
        while attempts_left > 0:
            outcome, items = _run_one_scan_attempt(cap)

            if outcome == "cancelled":
                print("\n  Item scanning cancelled.")
                break

            if outcome == "timeout":
                attempts_left -= 1
                if attempts_left > 0:
                    print(f"\n  No stable reading yet -- you have {attempts_left} more try.")
                    continue
                else:
                    print("\n  Still nothing detected after a second try -- ending this checkout.")
                    break

            # outcome == "stable" -- review before accepting
            decision = _show_summary_and_get_decision(cap, items)
            if decision == "accept":
                accepted_items = items
                print("\n  Items accepted:")
                for it in items:
                    print(f"     - {it['qty']}x {it['name']} @ N{it['price']}")
                break
            elif decision == "rescan":
                print("\n  Rescanning -- lay items out again.")
                continue  # doesn't consume a timeout attempt, this is the customer's own choice
            else:  # cancelled
                print("\n  Item scanning cancelled.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return accepted_items


def run_checkout(camera_index=None):
    # Pause entrance loop immediately when entering checkout
    pause_entrance()
    checkout_start_time = time.time()  # for session_duration, passed to save_transaction() below

    try:
        print("\n" + "=" * 56)
        print("   FLOWPOS -- CHECKOUT")
        print("=" * 56)

        # STEP 1: arrangement instructions, shown before anything else
        print("\nPlease arrange your items neatly and spaced apart,")
        print("so they don't overlap each other on camera.")

        # STEP 2: PIN entry -- verifying this is what triggers the camera next
        input_pin = input("\nPlease input your security PIN: ").strip()
        if not input_pin:
            print("Checkout cancelled: PIN input is required.")
            return False

        session = get_session_by_pin(input_pin)
        if not session:
            print(f"Invalid Security PIN '{input_pin}' or session expired.")
            return False

        customer_name = session["name"]
        staff_id = session["staff_id"]
        print(f"\nIdentity Verified: {customer_name} ({staff_id})")

        # STEP 3: camera source, cached after the first checkout of this run
        source = _get_checkout_camera_source()

        # STEP 4-6: scan (auto, with retry) -> review -> accept/rescan
        items = scan_items_horizontal(source)

        if not items:
            print("No items were confirmed. Checkout cancelled.")
            return False

        # STEP 7: finalize -- save the transaction, show the receipt, end session
        total_amount = sum(item["price"] * item["qty"] for item in items)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        session_duration = round(time.time() - checkout_start_time, 1)

        # save_transaction() is face_db.py's existing function (already used
        # by the System Info screen's "Total sales" figure) -- reused as-is,
        # not duplicated. It returns True/False, not an id; items round-trips
        # through json.dumps/loads unchanged, so passing our richer
        # {name, price, qty} list (instead of face_db's original plain
        # {name: qty} shape) preserves per-item price history without
        # needing any change to face_db.py itself.
        save_transaction(staff_id, customer_name, items, total_amount, session_duration)
        update_customer_stats(staff_id, total_amount)

        # Decrement recorded stock for each purchased item. add_stock() was
        # already written to support this ("pass a negative number to
        # decrement -- e.g. after a sale, if checkout is ever wired to call
        # this") -- just needed to actually be called. Deliberately not
        # clamped at zero: negative stock is a visible signal that
        # inventory has drifted from reality (miscounted item, manual
        # removal outside checkout, etc.) rather than something to hide.
        for item in items:
            add_stock(item["name"], -item["qty"])

        if _ON_EVENT:
            _ON_EVENT({"staff_id": staff_id, "name": customer_name,
                       "items": items, "total": total_amount})

        print("\n" + "=" * 56)
        print("           FLOWPOS -- RECEIPT")
        print("=" * 56)
        print(f" Customer : {customer_name} ({staff_id})")
        print(f" Time     : {timestamp}")
        print("-" * 56)
        print(f" {'ITEM NAME':<20}{'QTY':<6}{'PRICE':<10}{'TOTAL':<10}")
        print("-" * 56)
        for item in items:
            item_total = item["price"] * item["qty"]
            print(f" {item['name']:<20}{item['qty']:<6}N{item['price']:<9}N{item_total:<9}")
        print("=" * 56)
        print(f" TOTAL: N{total_amount}")
        print("=" * 56)
        print("\nSaved to transaction history -- Admin Panel > View Past Transactions any time.")

        input("\nPress Enter to end your session...")
        close_session(staff_id)
        print(f"\nSession ended for {customer_name}. Thank you for shopping with us!\n")
        return True

    finally:
        # Resume entrance background engine cleanly
        resume_entrance()


if __name__ == "__main__":
    run_checkout()