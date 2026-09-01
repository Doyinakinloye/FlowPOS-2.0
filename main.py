"""
SMART INVENTORY SYSTEM
Entry point — Central Menu with Controllable Entrance Daemon

UPDATED 2026-08-31 (entry flow rework) — Camera 1 now auto-starts the
moment main() runs, as part of the automatic startup sequence, rather
than waiting for a manual menu selection.

CORRECTED same day — an earlier version of this file also removed the
camera-source prompt entirely (always defaulting to the webcam, no
choice). That went further than asked: the requirement was "starts
automatically", not "no choice ever". Restored the get_stream_url()
prompt (phone IP Webcam vs. laptop webcam) -- it now just runs as
part of startup() instead of behind menu option 1. Camera 2
(checkout.py) already asks this same question every time it's used,
so no equivalent change was needed there.

FIXED 2026-09-01 — 'Q' and "Stop Entrance Engine" both silently did
nothing; only Ctrl+C actually worked. stop_entrance() here only ever
touched its OWN ENTRANCE_RUNNING flag, which the actual camera loop
in entrance.py was never even looking at -- so a genuinely running
loop never heard the stop signal. Now imports and calls
stop_entrance_engine() directly, and _loop() checks entrance.py's own
flag after each return to tell an intentional stop (don't restart)
from anything else, like a connection failure (still retries as
before). Verified with real background threads: a simulated 'Q'
press no longer triggers a retry, stop_entrance() now interrupts a
genuinely blocked fake loop, and a real connection-failure case still
retries same as always -- no regression there.
"""

import threading
import time
import os

from services.enrollment_manager import startup, show_system_info
from camera_stream import get_stream_url
from admin_panel import run_admin_panel

from enrollment.entrance import run_entrance, stop_entrance_engine
import enrollment.entrance as _entrance
from services.shopping_session import init_shopping_sessions_table

from checkout.checkout import run_checkout

# State variables to manage background execution
ENTRANCE_THREAD = None
ENTRANCE_RUNNING = False
ENTRANCE_STREAM_URL = None  # set by the get_stream_url() prompt on first start_entrance() call


def start_entrance():
    global ENTRANCE_THREAD, ENTRANCE_RUNNING, ENTRANCE_STREAM_URL

    if ENTRANCE_RUNNING:
        print("\n⚠️ Entrance engine is ALREADY running!")
        return

    if ENTRANCE_STREAM_URL is None:
        print("\n--- Camera 1: ENTRANCE (recognize / enroll) ---")
        ENTRANCE_STREAM_URL = get_stream_url()

    ENTRANCE_RUNNING = True

    def _loop():
        global ENTRANCE_RUNNING
        print("\n[Entrance Daemon] Camera 1 loop STARTED.")
        while ENTRANCE_RUNNING:
            try:
                run_entrance(ENTRANCE_STREAM_URL)
            except Exception as e:
                print(f"\n[Entrance Loop Exception]: {e}")
                time.sleep(1)
                continue
            # A normal return with entrance.py's OWN flag now False means
            # this was an intentional stop -- pressing 'Q' inside the
            # camera window, or stop_entrance() below calling
            # stop_entrance_engine(). Anything else (e.g. a camera that
            # failed to connect) leaves that flag True, so this still
            # retries exactly as before.
            if not _entrance.ENTRANCE_RUNNING:
                break
        ENTRANCE_RUNNING = False
        print("[Entrance Daemon] Camera 1 loop STOPPED cleanly.")

    ENTRANCE_THREAD = threading.Thread(target=_loop, daemon=True)
    ENTRANCE_THREAD.start()
    print("✅ Entrance camera started! Step up to Camera 1 to scan/enroll.")


def stop_entrance():
    global ENTRANCE_RUNNING
    if not ENTRANCE_RUNNING:
        print("\n⚠️ Entrance engine is not currently running.")
        return

    print("\nStopping entrance engine and releasing Camera 1...")
    ENTRANCE_RUNNING = False
    stop_entrance_engine()  # signals the ACTUAL running camera loop to exit -- setting
                            # only main.py's own flag above never reached it before
    time.sleep(1.5)  # Allow background iteration to terminate
    print("✅ Entrance engine stopped. Camera 1 freed for other operations.")


def main():
    print("\n" + "=" * 60)
    print("   SMART INVENTORY SYSTEM — Interactive Terminal")
    print("=" * 60)

    # Initialize system, SQLite schema, pre-load embeddings
    startup()
    init_shopping_sessions_table()

    # Camera 1 auto-starts -- no manual menu step needed.
    start_entrance()

    while True:
        status_str = "🟢 ACTIVE" if ENTRANCE_RUNNING else "🔴 STOPPED"
        print("\n" + "=" * 50)
        print(f"   MAIN MENU  (Entrance Status: {status_str})")
        print("=" * 50)
        print("ENTRANCE CONTROLS:")
        print("  1. Start Entrance Engine (Camera 1)")
        print("  2. Stop Entrance Engine  (Release Camera)")
        print()
        print("OPERATIONS:")
        print("  3. Run Checkout Flow (Camera 2 - Coming Next)")
        print()
        print("SYSTEM & MANAGEMENT:")
        print("  4. System Info")
        print("  5. Admin Panel (Shoppers, Database, Admin Tools)")
        print()
        print("  0. Exit")
        print("=" * 50)

        choice = input("Enter choice (0-5): ").strip()

        if choice == "1":
            start_entrance()
        elif choice == "2":
            stop_entrance()
        elif choice == '3':
            run_checkout()
        elif choice == "4":
            show_system_info()
        elif choice == "5":
            run_admin_panel()
        elif choice == "0":
            if ENTRANCE_RUNNING:
                stop_entrance()
            print("\nShutting down Smart Inventory System. Goodbye! 👋")
            break
        else:
            print("Invalid selection. Please choose an option from the menu.")


if __name__ == "__main__":
    main()