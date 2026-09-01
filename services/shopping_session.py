# ============================================
# SMART INVENTORY SYSTEM
# Module: Shopping Session Manager & Entrance Control
# Description: Session management + entrance loop pause/resume
# ============================================

import sqlite3
import random
import time

DB_PATH = "inventory.db"

# Global state flag for pausing entrance scanning
ENTRANCE_PAUSED = False


def pause_entrance():
    """Temporarily pauses entrance background scanning during checkout."""
    global ENTRANCE_PAUSED
    ENTRANCE_PAUSED = True
    print("\n  ⏸️ Entrance Engine temporarily PAUSED for checkout flow.")


def resume_entrance():
    """Resumes entrance background scanning after checkout completes."""
    global ENTRANCE_PAUSED
    ENTRANCE_PAUSED = False
    print("\n  ▶️ Entrance Engine RESUMED.")


def is_entrance_paused():
    """Checks if the entrance engine is currently paused."""
    return ENTRANCE_PAUSED


def _get_conn():
    return sqlite3.connect(DB_PATH)


def init_shopping_sessions_table():
    """Create the shopping_sessions table if it doesn't exist yet."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shopping_sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            pin           TEXT NOT NULL,
            customer_id   TEXT NOT NULL,
            customer_name TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'active',
            created_at    TEXT NOT NULL,
            closed_at     TEXT
        )
    ''')
    conn.commit()
    conn.close()


def _generate_unique_pin():
    """4-digit PIN not currently used by any ACTIVE session."""
    conn = _get_conn()
    cursor = conn.cursor()
    for _ in range(50):
        candidate = f"{random.randint(0, 9999):04d}"
        cursor.execute(
            "SELECT 1 FROM shopping_sessions WHERE pin = ? AND status = 'active'",
            (candidate,)
        )
        if cursor.fetchone() is None:
            conn.close()
            return candidate
    conn.close()
    raise RuntimeError("Could not generate a unique shopping PIN.")


def create_session(customer_id, customer_name):
    """Start a new shopping session and return 4-digit PIN."""
    init_shopping_sessions_table()
    pin = _generate_unique_pin()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO shopping_sessions
            (pin, customer_id, customer_name, status, created_at)
        VALUES (?, ?, ?, 'active', ?)
    ''', (pin, customer_id, customer_name, time.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return pin


def get_active_session_by_pin(pin):
    """Look up active session dictionary by PIN."""
    init_shopping_sessions_table()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, pin, customer_id, customer_name, created_at
        FROM shopping_sessions
        WHERE pin = ? AND status = 'active'
        ORDER BY id DESC LIMIT 1
    ''', (str(pin),))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "pin": row[1],
        "staff_id": row[2],
        "customer_id": row[2],
        "name": row[3],
        "customer_name": row[3],
        "created_at": row[4],
    }

get_session_by_pin = get_active_session_by_pin


def complete_checkout(session_id_or_staff_id):
    """Mark session as checked out (purchase completed)."""
    init_shopping_sessions_table()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE shopping_sessions
        SET status = 'checked_out', closed_at = ?
        WHERE (id = ? OR customer_id = ?) AND status = 'active'
    ''', (time.strftime("%Y-%m-%d %H:%M:%S"), session_id_or_staff_id, session_id_or_staff_id))
    conn.commit()
    conn.close()

close_session = complete_checkout
def list_active_sessions():
    """Retrieve all active shopping sessions from the database."""
    init_shopping_sessions_table()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, pin, customer_id, customer_name, created_at
        FROM shopping_sessions
        WHERE status = 'active'
        ORDER BY id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()

    sessions = []
    for row in rows:
        sessions.append({
            "id": row[0],
            "pin": row[1],
            "staff_id": row[2],
            "customer_id": row[2],
            "name": row[3],
            "customer_name": row[3],
            "created_at": row[4],
        })
    return sessions