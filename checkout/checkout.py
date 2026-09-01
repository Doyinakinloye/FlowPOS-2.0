# ============================================
# SMART INVENTORY SYSTEM
# Module: Checkout Flow
# Description: Automatic Pause/Resume of Entrance Engine during Checkout
#
# UPDATED 2026-08-31 — three changes from the version you had:
#   1. scan_items_horizontal() no longer returns a hardcoded fake list
#      (Coca-Cola Can x2, Lay's Chips x1). Pressing 'C' now runs the
#      real best.pt model on that frame, tallies detections by class
#      (coke/fanta/sprite/water), and prices them from
#      m4/price_catalog.py.
#   2. Currency display switched from "$" to "N" (Naira), matching
#      billing.py / excel_logger.py elsewhere in the project.
#   3. The "Camera 2 hardware not detected" fallback no longer invents
#      a fake purchase — that would have billed a real customer for
#      items that were never scanned. It now returns an empty list,
#      which run_checkout()'s existing "no items scanned" check
#      already handles safely (checkout is cancelled, nothing is
#      charged).
#
# Not yet tuned against real footage: CONF_THRESHOLD below. Detection
# runs on the single frame captured at the moment you press 'C' (not
# continuous tracking) — if items are getting missed, try lowering it
# or reposition within the guide rectangle.
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
from camera_stream import get_stream_url
from m4.price_catalog import get_price

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "best.pt"
)
CONF_THRESHOLD = 0.5  # detection confidence cutoff — tune against real footage

_MODEL = None


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
    Runs one detection pass on `frame`. Returns (scanned_items, annotated_frame)
    where scanned_items is in the same shape run_checkout() already expects:
    [{"name": label, "price": <N>, "qty": <count>}, ...]
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


def scan_items_horizontal(camera_index=1):
    """
    Connects to Camera 2 for item scanning. camera_index can be a
    physical device index (int), an IP Webcam stream URL (str,
    e.g. from get_stream_url()), or None/"" for the default webcam.
    """
    print("\n[Camera 2] Opening Item Scanning Camera...")
    source = _resolve_source(camera_index)
    cap = cv2.VideoCapture(source)

    # Fallback to device 0 only when the source was a plain device
    # index that failed to open — never fall back for a URL, since
    # trying device 0 instead of a phone stream would silently scan
    # the wrong camera.
    if not cap.isOpened() and isinstance(source, int):
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"⚠️ Camera 2 hardware not detected (source: {source!r}). Checkout cannot proceed without a real scan.")
        return []

    scanned_items = []
    scanning = True

    print("\n" + "=" * 56)
    print(" 📦 ITEM SCANNING ACTIVE")
    print(" Place all items horizontally on the counter below Camera 2.")
    print(" Press 'C' when ready to scan items, or 'Q' to cancel.")
    print("=" * 56)

    while scanning:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from Camera 2.")
            break

        H, W, _ = frame.shape

        cv2.rectangle(frame, (50, int(H * 0.4)), (W - 50, int(H * 0.85)), (0, 255, 255), 2)
        cv2.putText(frame, "LAY ITEMS HORIZONTALLY IN THIS AREA", (60, int(H * 0.38)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'C' to Scan | 'Q' to Cancel", (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Camera 2 - Item Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('c'), ord('C')):
            try:
                scanned_items, annotated = _detect_items(frame)
            except Exception as e:
                print(f"❌ Detection failed: {e}")
                break

            if not scanned_items:
                print("\n  ⚠️ No items detected — reposition and press 'C' again.")
                continue

            # Show what was detected for a moment before closing
            cv2.imshow("Camera 2 - Item Scanner", annotated)
            cv2.waitKey(1500)

            print("\n  ✅ Items scanned and identified successfully!")
            for item in scanned_items:
                print(f"     - {item['qty']}x {item['name']} @ N{item['price']}")
            scanning = False

        elif key in (ord('q'), ord('Q')):
            print("\n  ⚠️ Item scanning cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return scanned_items


def run_checkout(camera_index=None):
    # Pause entrance loop immediately when entering checkout
    pause_entrance()

    try:
        print("\n" + "=" * 56)
        print("   FLOWPOS — CHECKOUT & PAYMENT VERIFICATION")
        print("=" * 56)

        # STEP 1: PIN AUTHENTICATION
        input_pin = input("\n🔑 Please input your security PIN: ").strip()
        if not input_pin:
            print("❌ Checkout cancelled: PIN input is required.")
            return False

        session = get_session_by_pin(input_pin)
        if not session:
            print(f"❌ Invalid Security PIN '{input_pin}' or session expired.")
            return False

        customer_name = session["name"]
        staff_id = session["staff_id"]

        print(f"\n✅ Identity Verified: {customer_name} ({staff_id})")

        if camera_index is None:
            print("\n--- Camera 2: CHECKOUT (item scanning) ---")
            camera_index = get_stream_url()

        # STEP 2: HORIZONTAL ITEM PLACEMENT & SCANNING
        input("\n👉 Press Enter to open Camera 2 and scan your items...")
        items = scan_items_horizontal(camera_index)

        if not items:
            print("⚠️ No items were scanned. Checkout cancelled.")
            return False

        # STEP 3: ITEMIZED RECEIPT BREAKDOWN
        total_amount = sum(item["price"] * item["qty"] for item in items)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "=" * 56)
        print("           FLOWPOS — PROOF OF PAYMENT RECEIPT        ")
        print("=" * 56)
        print(f" Customer Name : {customer_name}")
        print(f" Customer ID   : {staff_id}")
        print(f" Security PIN  : {input_pin}")
        print(f" Transaction   : {timestamp}")
        print("-" * 56)
        print(f" {'ITEM NAME':<25} {'QTY':<8} {'PRICE':<10} {'TOTAL':<10}")
        print("-" * 56)

        for item in items:
            item_total = item["price"] * item["qty"]
            print(f" {item['name']:<25} {item['qty']:<8} N{item['price']:<9} N{item_total:<9}")

        print("=" * 56)
        print(f" GRAND TOTAL AMOUNT: N{total_amount}")
        print("=" * 56)

        # STEP 4: SCREENSHOT INSTRUCTION
        print("\n📸 PROOF OF PAYMENT INSTRUCTION:")
        print(f" -> Shopper '{customer_name}', please SCREENSHOT this terminal screen")
        print("    where your name and list appear, and attach it on your phone.")

        input("\nPress Enter once you have taken your screenshot to complete payment...")

        # STEP 5: CLOSE SESSION
        close_session(staff_id)
        print(f"\n🎉 Payment Complete & Verified! Session closed for {customer_name}.")
        print("   Thank you for shopping with us!\n")
        return True

    finally:
        # Resume entrance background engine cleanly
        resume_entrance()


if __name__ == "__main__":
    run_checkout()
