"""
sis_server.py
═════════════════════════════════════════════════════════════════════
Flask backend for the Smart Inventory System dashboard (index.html).

REWRITTEN to drive the CURRENT architecture (entrance.py + PIN-based
shopping_session.py) instead of the old tripwire pipeline (m2's
AccessController + station_bridge's StationBridge). Camera 2 / item
detection / billing is NOT wired here yet — that's the Checkout flow,
which hasn't been built. This server auto-billing simply stays off
until then; item detection is preview-only.

Key mechanism: EntranceEngine runs run_entrance(stream_url) in a loop
on a background thread — the exact same pattern main.py's
_entrance_loop uses. Its cv2.imshow calls are redirected into the
browser (same trick as before, now matching "Entrance"/"Enrollment"
window titles instead of "Recognition"/"Item Detection").

The one wrinkle: run_enrollment() (called automatically by
run_entrance() when someone isn't recognized) uses plain input() for
its name/retry prompts — this is a real terminal interaction, and a
web server has no terminal. Rather than touch entrance.py or
enrollment.py at all, this file monkey-patches builtins.input the
same way it already monkey-patches cv2 functions: when input() is
called from a BACKGROUND thread (never the Flask request thread), it
pauses that thread and waits for the browser to POST an answer via
/api/enrollment/answer. Flask itself never blocks.

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
    rename_unknown_customer,
)
from embedding_loader import remove_embedding_from_memory
from enrollment.entrance import run_entrance
from services.shopping_session import (
    init_shopping_sessions_table,
    list_active_sessions,
)
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
#  entrance.py and enrollment.py's real windows ("Smart Inventory -
#  Entrance" / "Smart Inventory - Enrollment") get routed to the
#  dashboard instead of a native popup. No changes to either file —
#  we only intercept the OpenCV calls they already make.
# ════════════════════════════════════════════════════════════════
_PIPELINE_STOP = threading.Event()

def _cv2_imshow(winname, mat):
    w = str(winname)
    jpg = _encode(mat)
    if jpg is None:
        return
    if "Entrance" in w or "Enrollment" in w or "Recognition" in w:
        STATE.recog_jpeg = jpg
        STATE.recog_online = True

def _cv2_waitkey(delay=1):
    try:
        if delay and delay > 0:
            time.sleep(min(int(delay), 30) / 1000.0)
    except Exception:
        pass
    return ord('q') if _PIPELINE_STOP.is_set() else -1

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
#  INPUT BROKER
#  run_enrollment() (called automatically by run_entrance() when
#  someone isn't recognized) uses plain input() for its name/retry
#  prompts. This intercepts input() ONLY when called from a
#  background thread (never Flask's own request thread) and routes
#  the prompt to the browser, blocking that background thread until
#  /api/enrollment/answer is POSTed. Flask stays fully responsive.
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
                result = run_entrance(stream_url)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[server] entrance loop error: {e}")
                time.sleep(2)
                continue
            self._record_result(result)
        STATE.recog_online = False
        print("[server] Entrance loop stopped.")

    def _record_result(self, result):
        with STATE.lock:
            if result is None:
                _push(STATE.recognition_log, {
                    "timestamp": _now(), "event": "NO_CHECKIN",
                    "name": None, "customer_id": None,
                    "confidence": None, "pin": None,
                })
            else:
                ev = "NEW_CUSTOMER_ENROLLED" if result.get("is_new") else "CUSTOMER_IDENTIFIED"
                _push(STATE.recognition_log, {
                    "timestamp": _now(), "event": ev,
                    "name": result.get("name"),
                    "customer_id": result.get("staff_id"),
                    "confidence": None,
                    "pin": result.get("pin"),
                })
        STATE.events_fired += 1

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
    return jsonify({
        "enrolled_customers": enrolled,
        "unknown_captures": unknown,   # stays 0 under the current architecture
        "active_sessions": len(sessions),
        # Transactions/revenue have no data source until Checkout (Camera 2)
        # is built — left at 0 rather than faked.
        "transactions_today": 0,
        "revenue_today": 0,
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
        # Auto-billing doesn't exist yet — no Checkout flow built.
        "auto_billing": False,
    })


# ── enrollment status / input broker ──────────────────────────────
@app.route("/api/enrollment/next-id")
def api_next_id():
    try:
        return jsonify({"staff_id": generate_staff_id()})
    except Exception:
        return jsonify({"staff_id": ""})


@app.route("/api/enrollment/status")
def api_enroll_status():
    """
    Polled by the dashboard. When the automatic walk-up flow hits an
    unrecognized face, run_enrollment() will be mid-input() waiting
    on the browser — this surfaces that prompt so the UI can show a
    text box and let the person type their name (or answer a retry
    prompt) without ever touching a terminal.
    """
    return jsonify({
        "waiting_for_input": _INPUT_STATE["waiting"],
        "prompt": _INPUT_STATE["prompt"],
    })


@app.route("/api/enrollment/answer", methods=["POST"])
def api_enroll_answer():
    data = request.get_json(silent=True) or {}
    answer = data.get("answer", "")
    if not _INPUT_STATE["waiting"]:
        return jsonify({"ok": False, "error": "Nothing is waiting for input right now."}), 409
    _INPUT_STATE["answer"] = answer
    _INPUT_EVENT.set()
    return jsonify({"ok": True})


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
        "object_detection": "Not yet active — Checkout (Camera 2) not built",
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