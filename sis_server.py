"""
sis_server.py
═════════════════════════════════════════════════════════════════════
Flask backend for the FlowPOS 2.0 web UI.

REWRITTEN to drive the CURRENT architecture (entrance.py + PIN-based
shopping_session.py) instead of the old tripwire pipeline (m2's
AccessController + station_bridge's StationBridge).

UPDATED 2026-09-02 -- Checkout (Camera 2) is now wired in too, using
the exact same "don't touch the real, hardware-tested module" approach
already used for entrance.py:
  - checkout.py's window ("Camera 2 - Item Scanner") is routed into
    STATE.det_jpeg by the same cv2.imshow redirect entrance.py uses,
    just matching a different window-name substring.
  - checkout.py's PIN prompt AND its own camera-source prompt (both
    plain input() calls) are picked up automatically by the SAME
    generic input broker built for enrollment's name prompt -- it
    doesn't care what the prompt text is, only that input() was
    called from a background thread. No new code was needed for this
    part, just running checkout in one.
  - checkout.py's interactive keys ('C'/'A'/'R'/'Q') needed something
    genuinely new though: the old cv2.waitKey redirect only ever
    returned "no key" or a forced stop, with no way to tell it "the
    Accept button was clicked." Added a small _PENDING_KEY relay for
    this -- POST a key to /api/checkout/key, the next cv2.waitKey()
    call inside checkout.py returns it once, then goes back to "no
    key" until another one arrives. Confirmed safe to share globally
    (not scoped per-engine) because checkout.py always runs with
    entrance PAUSED (pause_entrance()/is_entrance_paused()) -- so
    entrance is never also calling cv2.waitKey() at the same time.
  - Both entrance.py and checkout.py now have an optional event hook
    (set_event_hook()) added specifically for this dashboard -- fires
    once per real event (a PIN issued, a sale completed) rather than
    depending on either function's return value, which would have
    required them to stop and restart their loops far more often than
    is safe for the terminal experience. See each file's own comments
    for why. This REPLACES the old _record_result()-on-return-value
    approach below, which could only ever fire once per engine
    start/stop (not per customer), regardless of what run_entrance()
    returned -- a pre-existing gap, not something this update broke.

UPDATED 2026-09-02 (later same day) -- added three inventory-editing
routes (add-stock/set-stock/set-price). Without them, "the admin never
needs the terminal" was false specifically for inventory management --
the web UI could view stock but never actually change it. Each just
wraps the same, already-tested services/inventory.py functions the
terminal admin panel uses.

UPDATED 2026-09-03 -- checkout.py has its OWN separate camera-source
cache (_CHECKOUT_STREAM_SOURCE), asked once via a hidden terminal-style
prompt and never touchable from the browser -- completely disconnected
from the "Camera 2 -- Checkout" field on the Cameras page, which only
ever affected entrance/preview. CheckoutEngine.start() now pre-seeds
checkout.py's cache from STATE.detection_cam (the Cameras page value)
right before starting, whenever the admin has actually set one.

FIXED 2026-09-03 (same day) -- real bug, found from screen recordings
and confirmed by direct reproduction with checkout's own unmodified
scanning code: _PIPELINE_STOP is meant ONLY to let the entrance engine's
Stop button interrupt a blocked cv2.waitKey() call. But cv2.waitKey is
patched globally, so checkout.py's own scanning loop was ALSO reading
it -- and treating a stale "entrance was stopped" signal exactly like a
real 'Q' keypress. If entrance had ever been stopped and not restarted
before a checkout began, the very first waitKey() call inside checkout's
scan loop would return 'q' and cancel instantly -- after showing exactly
one real frame with real detections, then reverting. That matches the
reported symptom precisely: camera opens, items are visible, gone in a
blink. Fixed by only honoring _PIPELINE_STOP when CHECKOUT_ENGINE isn't
running -- entrance is always fully paused (not calling cv2.waitKey at
all) during an active checkout, so this changes nothing for entrance's
own stop behavior. Verified three ways: reproduced the failure with
checkout's real _run_one_scan_attempt() before the fix, confirmed it no
longer fails after the fix (same poisoned state), and confirmed entrance's
own stop signal still works correctly when checkout isn't running.

The UI itself is never expected to need a terminal or a keyboard --
every interaction (PIN entry, name entry, Accept/Rescan/Cancel) is a
real button or text box in the browser; this file's job is entirely
the translation layer connecting web actions to the same, unmodified
recognize/enroll/checkout code already tested on real hardware.

Run it from the project root (same folder as main.py):
    python sis_server.py
then open  http://localhost:5000  in your browser.
═════════════════════════════════════════════════════════════════════
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import time
import base64
import threading
import builtins
import datetime

import cv2
import logging
from flask import Flask, jsonify, request, send_from_directory, send_file

logging.getLogger("werkzeug").setLevel(logging.ERROR)

try:
    from flask_cors import CORS
    _HAS_CORS = True
except Exception:
    _HAS_CORS = False

# ── Current-architecture project modules ──
from services.enrollment_manager import startup, generate_staff_id
from camera_stream import CameraStream
from face_db import (
    get_all_customers, delete_customer, customer_exists,
    rename_unknown_customer, get_all_transactions,
)
from embedding_loader import remove_embedding_from_memory
from enrollment.entrance import run_entrance
import enrollment.entrance as entrance
import checkout.checkout as checkout_flow
from services.shopping_session import (
    init_shopping_sessions_table,
    list_active_sessions,
)
from services.inventory import get_inventory, add_stock, set_stock_quantity, set_price
from m4.price_catalog import PRICES   # display-only, for the System Info / catalog panel

HERE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=HERE)
if _HAS_CORS:
    CORS(app)


# ════════════════════════════════════════════════════════════════
#  SHARED STATE
# ════════════════════════════════════════════════════════════════
class State:
    def __init__(self):
        self.lock = threading.Lock()

        # Camera config from the Cameras page.
        # ""  -> laptop webcam 0/1, "webcam:N" -> index N, else a stream URL.
        self.recognition_cam = ""     # entrance/face camera (CAM-01)
        self.detection_cam   = ""     # shelf camera (CAM-02) — preview only, no billing yet

        self.recog_jpeg = None
        self.det_jpeg   = None
        self.recog_online = False
        self.det_online   = False

        # Every entrance.py cycle outcome gets pushed here.
        self.recognition_log = []     # {timestamp,event,name,customer_id,confidence,pin}
        self.events_fired = 0


STATE = State()
LOG_CAP = 200


def _push(buf, item):
    buf.append(item)
    if len(buf) > LOG_CAP:
        del buf[: len(buf) - LOG_CAP]


def _now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _resolve_cam(value, default_index):
    """
    ""  / None   -> laptop webcam at `default_index`
    "webcam:N"   -> laptop webcam index N
    "http://..." -> phone-camera URL, used as-is
    """
    if value is None:
        return default_index
    value = str(value).strip()
    if value == "":
        return default_index
    if value.startswith("webcam:"):
        try:
            return int(value.split(":", 1)[1])
        except Exception:
            return default_index
    return value


# ════════════════════════════════════════════════════════════════
#  cv2 → browser redirect
#  entrance.py, enrollment.py, and checkout.py's real windows
#  ("Smart Inventory - Entrance" / "Smart Inventory - Enrollment" /
#  "Camera 2 - Item Scanner") get routed to the dashboard instead of
#  a native popup. No changes to any of those files — we only
#  intercept the OpenCV calls they already make.
# ════════════════════════════════════════════════════════════════
_PIPELINE_STOP = threading.Event()

# Set via /api/checkout/key -- consumed exactly once by the next
# cv2.waitKey() call, then goes back to "no key". Shared globally
# rather than scoped per-engine: safe because checkout.py always runs
# with entrance PAUSED (pause_entrance()/is_entrance_paused()), so
# entrance is never also calling cv2.waitKey() while a checkout key
# is pending -- there's no other consumer to steal it.
_PENDING_KEY = None

def _cv2_imshow(winname, mat):
    w = str(winname)
    jpg = _encode(mat)
    if jpg is None:
        return
    if "Entrance" in w or "Enrollment" in w or "Recognition" in w:
        STATE.recog_jpeg = jpg
        STATE.recog_online = True
    elif "Checkout" in w or "Item Scanner" in w:
        STATE.det_jpeg = jpg
        STATE.det_online = True

def _cv2_waitkey(delay=1):
    global _PENDING_KEY
    try:
        if delay and delay > 0:
            time.sleep(min(int(delay), 30) / 1000.0)
    except Exception:
        pass
    if _PENDING_KEY is not None:
        k = _PENDING_KEY
        _PENDING_KEY = None
        return k
    # _PIPELINE_STOP belongs to the entrance engine ONLY -- but cv2.waitKey
    # itself is patched globally, so checkout.py's own scanning loop was
    # ALSO seeing it, and treating it exactly like a real 'Q' keypress.
    # Confirmed by direct reproduction: if entrance was ever stopped and
    # never restarted before a checkout began, _PIPELINE_STOP stays set,
    # and checkout's real _run_one_scan_attempt() returns "cancelled"
    # on its very first loop -- after showing exactly one real frame with
    # real detections, then reverting instantly. That matches the report
    # precisely: camera opens, items are visible, then it's gone in a
    # blink. Entrance is always fully paused (not calling cv2.waitKey at
    # all) while a checkout is running, so it's safe to simply not apply
    # this signal during that window -- only an explicit checkout key
    # (handled above) should ever be able to end a checkout.
    if _PIPELINE_STOP.is_set() and not CHECKOUT_ENGINE.running:
        return ord('q')
    return -1

def _cv2_noop(*a, **k):
    return None

def _install_cv2_redirect():
    cv2.imshow            = _cv2_imshow
    cv2.waitKey           = _cv2_waitkey
    cv2.namedWindow       = _cv2_noop
    cv2.destroyWindow     = _cv2_noop
    cv2.destroyAllWindows = _cv2_noop
    cv2.setWindowProperty = _cv2_noop
    cv2.moveWindow        = _cv2_noop
    cv2.resizeWindow      = _cv2_noop


def _encode(frame):
    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        return base64.b64encode(buf).decode("ascii")
    except Exception:
        return None


_install_cv2_redirect()


# ════════════════════════════════════════════════════════════════
#  INPUT BROKER (generic -- not enrollment-specific despite the
#  original comment below; also used by checkout.py's PIN prompt and
#  its own camera-source prompt)
#  run_enrollment() (called automatically by run_entrance() when
#  someone isn't recognized) uses plain input() for its name/retry
#  prompts. This intercepts input() ONLY when called from a
#  background thread (never Flask's own request thread) and routes
#  the prompt to the browser, blocking that background thread until
#  /api/input/answer is POSTed. Flask stays fully responsive.
# ════════════════════════════════════════════════════════════════
_INPUT_STATE = {"waiting": False, "prompt": ""}
_INPUT_EVENT = threading.Event()
_orig_input  = builtins.input

def _patched_input(prompt=""):
    if threading.current_thread() is threading.main_thread():
        return _orig_input(prompt)
    _INPUT_EVENT.clear()
    _INPUT_STATE["waiting"] = True
    _INPUT_STATE["prompt"]  = str(prompt)
    _INPUT_STATE["answer"]  = None
    print(f"[server] Waiting for browser input: {prompt!r}")
    _INPUT_EVENT.wait()
    answer = _INPUT_STATE.get("answer") or ""
    _INPUT_STATE["waiting"] = False
    _INPUT_STATE["prompt"]  = ""
    return answer

builtins.input = _patched_input


# ════════════════════════════════════════════════════════════════
#  ENTRANCE ENGINE — Camera 1. Real recognize-or-enroll pipeline.
# ════════════════════════════════════════════════════════════════
class EntranceEngine:
    def __init__(self):
        self.lock    = threading.Lock()
        self.running = False
        self._thread = None

    def start(self):
        with self.lock:
            if self.running:
                return False, "Entrance is already running."

            _PIPELINE_STOP.clear()
            recog_src = _resolve_cam(STATE.recognition_cam, 0)

            PREVIEW.stop()
            STATE.recog_jpeg = None

            print("")
            print("=" * 56)
            print("[server] STARTING ENTRANCE ENGINE (real pipeline)")
            print(f"[server] Camera 1 (entrance): {recog_src!r}")
            print("=" * 56)

            self.running = True
            self._thread = threading.Thread(
                target=self._loop, args=(recog_src,), daemon=True)
            self._thread.start()
            return True, "Entrance engine started."

    def _loop(self, stream_url):
        while self.running and not _PIPELINE_STOP.is_set():
            try:
                run_entrance(stream_url)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[server] entrance loop error: {e}")
                time.sleep(2)
                continue
        STATE.recog_online = False
        print("[server] Entrance loop stopped.")

    def stop(self):
        with self.lock:
            if not self.running:
                return False
            self.running = False
        _PIPELINE_STOP.set()
        # Also release the input broker if something was mid-prompt,
        # so the background thread doesn't hang forever waiting.
        if _INPUT_STATE["waiting"]:
            _INPUT_STATE["answer"] = ""
            _INPUT_EVENT.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8)
        STATE.recog_online = False
        STATE.recog_jpeg = None
        if STATE.recognition_cam or STATE.detection_cam:
            PREVIEW.start()
        return True


ENGINE = EntranceEngine()


# ════════════════════════════════════════════════════════════════
#  PREVIEW  (raw camera feed while the engine is NOT running —
#  architecture-agnostic, reused as-is)
# ════════════════════════════════════════════════════════════════
class PreviewStreamer:
    def __init__(self):
        self._threads = []
        self._stop = threading.Event()
        self.running = False

    def start(self):
        self.stop()
        self._stop = threading.Event()
        self.running = True
        recog_src = _resolve_cam(STATE.recognition_cam, 0)
        det_src   = _resolve_cam(STATE.detection_cam, 0)
        print(f"[server] Preview ON — cam1={recog_src!r} cam2={det_src!r}")
        self._threads = [
            threading.Thread(target=self._stream, args=(recog_src, "recog"), daemon=True),
            threading.Thread(target=self._stream, args=(det_src, "det"), daemon=True),
        ]
        for t in self._threads:
            t.start()

    def _stream(self, src, which):
        cam = CameraStream(stream_url=src)
        if not cam.connect():
            if which == "recog":
                STATE.recog_online = False
            else:
                STATE.det_online = False
            return
        while not self._stop.is_set():
            frame = cam.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            jpg = _encode(frame)
            if which == "recog":
                STATE.recog_jpeg = jpg
                STATE.recog_online = True
            else:
                STATE.det_jpeg = jpg
                STATE.det_online = True
            time.sleep(0.04)
        cam.release()
        if which == "recog":
            STATE.recog_online = False
        else:
            STATE.det_online = False

    def stop(self):
        if not self.running and not self._threads:
            return
        self._stop.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=3)
        self._threads = []
        self.running = False
        time.sleep(0.3)


PREVIEW = PreviewStreamer()


# ════════════════════════════════════════════════════════════════
#  EVENT HOOKS — real per-event dashboard logging.
#  Registered once at startup (bottom of this file). Each fires
#  exactly once per real event, straight from entrance.py/checkout.py
#  themselves -- see this file's top docstring for why this replaced
#  the old return-value approach.
# ════════════════════════════════════════════════════════════════
def _on_entrance_event(ev):
    with STATE.lock:
        event_type = "NEW_CUSTOMER_ENROLLED" if ev.get("is_new") else "CUSTOMER_IDENTIFIED"
        _push(STATE.recognition_log, {
            "timestamp": _now(), "event": event_type,
            "name": ev.get("name"), "customer_id": ev.get("staff_id"),
            "confidence": None, "pin": ev.get("pin"),
        })
    STATE.events_fired += 1


def _on_checkout_event(ev):
    with STATE.lock:
        _push(STATE.recognition_log, {
            "timestamp": _now(), "event": "CHECKOUT_COMPLETE",
            "name": ev.get("name"), "customer_id": ev.get("staff_id"),
            "confidence": None, "pin": None,
            "items": ev.get("items"), "total": ev.get("total"),
        })
    STATE.events_fired += 1


# ════════════════════════════════════════════════════════════════
#  CHECKOUT ENGINE — Camera 2. Runs the real, unmodified
#  checkout.run_checkout() in a background thread. Its PIN prompt and
#  its own camera-source prompt are both picked up automatically by
#  the generic input broker above -- no special-casing needed here,
#  the broker doesn't care what the prompt text is.
# ════════════════════════════════════════════════════════════════
class CheckoutEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.last_result = None
        self._thread = None

    def start(self):
        with self.lock:
            if self.running:
                return False, "A checkout is already in progress."

            # If the admin has set a Camera 2 address on the Cameras
            # page, hand it to checkout.py's own camera-source cache
            # right now, before anything else runs. checkout.py itself
            # is completely untouched by this -- it still just checks
            # "is my cache empty?" the same way it always has; this
            # only makes sure that check finds something real instead
            # of a stale value from whenever it first got asked. If
            # the admin hasn't set anything here, this does nothing,
            # and checkout falls back to its own original one-time
            # prompt exactly as before.
            if STATE.detection_cam:
                checkout_flow._CHECKOUT_STREAM_SOURCE = _resolve_cam(STATE.detection_cam, 1)

            self.running = True
            self.last_result = None
            STATE.det_jpeg = None
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True, "Checkout started."

    def _run(self):
        try:
            self.last_result = checkout_flow.run_checkout()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[server] checkout error: {e}")
            self.last_result = False
        finally:
            with self.lock:
                self.running = False
            STATE.det_online = False


CHECKOUT_ENGINE = CheckoutEngine()
entrance.set_event_hook(_on_entrance_event)
checkout_flow.set_event_hook(_on_checkout_event)


# ════════════════════════════════════════════════════════════════
#  ROUTES — page
# ════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


# ── customers ────────────────────────────────────────────────────
def _customer_view(c):
    sid = c["staff_id"]
    return {
        "name": c["name"],
        "staff_id": sid,
        "enrolled_at": c.get("enrolled_at"),
        "total_purchases": c.get("total_purchases") or 0,
        "total_spent": c.get("total_spent") or 0,
        # entrance.py never creates UNK- customers (unlike the old m2
        # pipeline) — it either recognizes someone or runs the full
        # named enrollment. This field stays here for UI compatibility
        # but will always be False under the current architecture.
        "is_unknown": str(sid).startswith("UNK-"),
        "has_embedding": c.get("embedding") is not None,
    }


@app.route("/api/customers")
def api_customers():
    return jsonify([_customer_view(c) for c in get_all_customers()])


@app.route("/api/customers/<staff_id>", methods=["DELETE"])
def api_delete_customer(staff_id):
    if not customer_exists(staff_id):
        return jsonify({"ok": False, "error": "Not found"}), 404

    folder = None
    for c in get_all_customers():
        if c["staff_id"] == staff_id:
            folder = c.get("folder_path")
            break

    ok = delete_customer(staff_id)
    if ok:
        try:
            remove_embedding_from_memory(staff_id)
        except Exception as e:
            print(f"[server] remove_embedding_from_memory: {e}")
        try:
            if folder and os.path.isdir(folder):
                import shutil; shutil.rmtree(folder, ignore_errors=True)
        except Exception as e:
            print(f"[server] folder delete: {e}")
        print(f"[server] Deleted customer {staff_id}")
    return jsonify({"ok": bool(ok)})


@app.route("/api/customers/<staff_id>/image")
def api_customer_image(staff_id):
    folder = None
    for c in get_all_customers():
        if c["staff_id"] == staff_id:
            folder = c.get("folder_path")
            break
    if folder and os.path.isdir(folder):
        cand = os.path.join(folder, "best_frame.jpg")
        if not os.path.isfile(cand):
            imgs = sorted(f for f in os.listdir(folder)
                          if f.lower().endswith((".jpg", ".jpeg", ".png")))
            cand = os.path.join(folder, imgs[0]) if imgs else None
        if cand and os.path.isfile(cand):
            return send_file(cand, mimetype="image/jpeg")
    return ("", 404)


@app.route("/api/customers/<staff_id>/rename", methods=["POST"])
def api_rename_customer(staff_id):
    data = request.get_json(silent=True) or {}
    name   = (data.get("name") or "").strip()
    new_id = (data.get("new_id") or "").strip() or staff_id
    if not name:
        return jsonify({"ok": False, "error": "Name required"}), 400
    ok = rename_unknown_customer(staff_id, name, new_id)
    return jsonify({"ok": bool(ok)})


# ── stats / sessions / logs ──────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    customers = get_all_customers()
    enrolled = sum(1 for c in customers if not str(c["staff_id"]).startswith("UNK-"))
    unknown  = sum(1 for c in customers if str(c["staff_id"]).startswith("UNK-"))
    sessions = list_active_sessions()

    # Real numbers now that checkout is wired in -- filtered to
    # today's date from each transaction's own timestamp.
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    todays_txns = [t for t in get_all_transactions() if t.get("timestamp", "").startswith(today_str)]

    return jsonify({
        "enrolled_customers": enrolled,
        "unknown_captures": unknown,   # stays 0 under the current architecture
        "active_sessions": len(sessions),
        "transactions_today": len(todays_txns),
        "revenue_today": sum(t.get("total", 0) for t in todays_txns),
        "cameras": {
            "recognition": STATE.recog_online,
            "detection": STATE.det_online,
        },
    })


@app.route("/api/sessions")
def api_sessions():
    """
    PIN-based active shopping sessions from shopping_session.py.
    No cart/total yet — Checkout (Camera 2) hasn't been built, so
    there's no item data to attach to a session. The 'pin' field is
    what the UI should surface prominently; 'cart'/'total' are kept
    at empty/0 for compatibility with the existing dashboard cards.
    """
    sessions = list_active_sessions()
    return jsonify([{
        "name": s["customer_name"],
        "staff_id": s["customer_id"],
        "pin": s["pin"],
        "start_time": s["created_at"].split(" ")[-1] if " " in s["created_at"] else s["created_at"],
        "is_unknown": False,
        "total": 0,
        "cart": {},
    } for s in sessions])


@app.route("/api/recognition/log")
def api_rec_log():
    with STATE.lock:
        return jsonify(list(STATE.recognition_log))


# ── camera frames + status ───────────────────────────────────────
@app.route("/api/camera/recognition/frame")
def api_recog_frame():
    return jsonify({"frame": STATE.recog_jpeg, "online": STATE.recog_online})


@app.route("/api/camera/detection/frame")
def api_det_frame():
    return jsonify({"frame": STATE.det_jpeg, "online": STATE.det_online})


@app.route("/api/camera/status")
def api_cam_status():
    return jsonify({
        "recognition": {"online": STATE.recog_online},
        "detection":   {"online": STATE.det_online},
    })


@app.route("/api/config/cameras", methods=["POST"])
def api_config_cameras():
    """
    Update camera sources. If the entrance engine is running, it's
    restarted so the new source takes effect immediately. If idle,
    the live preview starts/restarts instead.
    """
    data = request.get_json(silent=True) or {}
    changed = False
    if "recognition" in data:
        STATE.recognition_cam = data.get("recognition") or ""
        changed = True
    if "detection" in data:
        STATE.detection_cam = data.get("detection") or ""
        changed = True

    if changed and ENGINE.running:
        ENGINE.stop()
        ok, msg = ENGINE.start()
        return jsonify({"ok": ok, "message": msg})

    if STATE.recognition_cam or STATE.detection_cam:
        PREVIEW.start()
    else:
        PREVIEW.stop()
    return jsonify({"ok": True})


# ── engine control (Start/Stop Recognition button) ────────────────
@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    ok, msg = ENGINE.start()
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 409)


@app.route("/api/pipeline/stop", methods=["POST"])
def api_pipeline_stop():
    ENGINE.stop()
    return jsonify({"ok": True})


@app.route("/api/pipeline/status")
def api_pipeline_status():
    return jsonify({
        "running": ENGINE.running,
        "checkout_running": CHECKOUT_ENGINE.running,
    })


# ── enrollment status / input broker ──────────────────────────────
@app.route("/api/enrollment/next-id")
def api_next_id():
    try:
        return jsonify({"staff_id": generate_staff_id()})
    except Exception:
        return jsonify({"staff_id": ""})


@app.route("/api/input/status")
def api_input_status():
    """
    Polled by the dashboard. This is the GENERIC input broker status --
    not enrollment-specific despite living near the enrollment routes.
    Anything running in a background thread that calls input() shows
    up here: enrollment's name prompt, checkout's PIN prompt, and
    checkout's own camera-source prompt (its first use only, before
    the source gets cached) all flow through this same mechanism.
    The UI should show a text box for whatever "prompt" says, without
    needing to know which of those three it actually is.
    """
    return jsonify({
        "waiting_for_input": _INPUT_STATE["waiting"],
        "prompt": _INPUT_STATE["prompt"],
    })


@app.route("/api/input/answer", methods=["POST"])
def api_input_answer():
    data = request.get_json(silent=True) or {}
    answer = data.get("answer", "")
    if not _INPUT_STATE["waiting"]:
        return jsonify({"ok": False, "error": "Nothing is waiting for input right now."}), 409
    _INPUT_STATE["answer"] = answer
    _INPUT_EVENT.set()
    return jsonify({"ok": True})


# ── checkout (Camera 2) ────────────────────────────────────────────
@app.route("/api/checkout/start", methods=["POST"])
def api_checkout_start():
    ok, msg = CHECKOUT_ENGINE.start()
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 409)


@app.route("/api/checkout/status")
def api_checkout_status():
    """
    Polled while a checkout is in progress. 'last_result' is None
    while running, then True/False once it finishes -- the UI can use
    that to show a brief confirmation before returning to idle.
    """
    return jsonify({
        "running": CHECKOUT_ENGINE.running,
        "last_result": CHECKOUT_ENGINE.last_result,
    })


@app.route("/api/checkout/key", methods=["POST"])
def api_checkout_key():
    """
    Translates a browser button click into the keypress checkout.py
    is already listening for on real hardware -- 'c' force-confirms a
    scan, 'a' accepts and pays, 'r' rescans, 'q' cancels. The UI never
    needs to show or know about any of these letters; each maps to
    one clearly-labeled button.
    """
    data = request.get_json(silent=True) or {}
    key = (data.get("key") or "").strip().lower()
    if key not in ("c", "a", "r", "q"):
        return jsonify({"ok": False, "error": "Invalid key"}), 400
    global _PENDING_KEY
    _PENDING_KEY = ord(key)
    return jsonify({"ok": True})


# ── transactions / inventory ────────────────────────────────────────
@app.route("/api/transactions")
def api_transactions():
    return jsonify(get_all_transactions())


@app.route("/api/inventory")
def api_inventory():
    return jsonify(get_inventory())


@app.route("/api/inventory/add-stock", methods=["POST"])
def api_inventory_add_stock():
    data = request.get_json(silent=True) or {}
    name = (data.get("item_name") or "").strip().lower()
    try:
        qty = int(data.get("quantity"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Quantity must be a whole number."}), 400
    ok = add_stock(name, qty)
    return jsonify({"ok": bool(ok)}), (200 if ok else 404)


@app.route("/api/inventory/set-stock", methods=["POST"])
def api_inventory_set_stock():
    data = request.get_json(silent=True) or {}
    name = (data.get("item_name") or "").strip().lower()
    try:
        qty = int(data.get("quantity"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Quantity must be a whole number."}), 400
    ok = set_stock_quantity(name, qty)
    return jsonify({"ok": bool(ok)}), (200 if ok else 404)


@app.route("/api/inventory/set-price", methods=["POST"])
def api_inventory_set_price():
    data = request.get_json(silent=True) or {}
    name = (data.get("item_name") or "").strip().lower()
    try:
        price = float(data.get("price"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Price must be a number."}), 400
    ok = set_price(name, price)
    return jsonify({"ok": bool(ok)}), (200 if ok else 404)


# ── system info ──────────────────────────────────────────────────
@app.route("/api/system/info")
def api_system_info():
    recog = STATE.recognition_cam or "Laptop webcam (0)"
    det   = STATE.detection_cam or "Laptop webcam (1)"
    return jsonify({
        "recognition_cam": recog,
        "detection_cam": det,
        "face_detection": "MediaPipe (via enrollment.py's shared pipeline)",
        "face_recognition": "DeepFace · ArcFace (centroid match, entrance.py)",
        "object_detection": "YOLOv8 (best.pt, via checkout.py)",
        "enrollment_poses": ["Straight", "Left", "Right", "Up", "Down"],
        "recognition_threshold": None,  # not runtime-adjustable under entrance.py yet
        "database": "SQLite (inventory.db)",
        "currency": "\u20a6 (Naira)",
        "price_catalog": dict(PRICES),
    })


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("   SMART INVENTORY SYSTEM — Web Server")
    print("=" * 60)
    try:
        startup()                          # loads embeddings / DB, like main.py
        init_shopping_sessions_table()     # was never called anywhere before this
    except Exception as e:
        print(f"[server] startup() warning: {e}")
    print("\nDashboard:  http://localhost:5000")
    print("Set your camera URLs on the Cameras page, then press")
    print("'Start Recognition'.  Ctrl+C here to stop the server.\n")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)