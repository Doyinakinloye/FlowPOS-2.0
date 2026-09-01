# ============================================
# SMART INVENTORY SYSTEM
# Module: Inventory Management
#
# NEW 2026-08-31 — added for the admin panel's "Check Inventory"
# option (view items/quantity/price, update stock). Lives in the same
# inventory.db that face_db.py and shopping_session.py already use
# (DB_PATH = "inventory.db"), not the separate legacy smart_store.db.
# ============================================

import sqlite3
import time

DB_PATH = "inventory.db"


def _get_conn():
    return sqlite3.connect(DB_PATH)


def init_inventory_table():
    """Create the inventory table if it doesn't exist yet."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            item_name      TEXT PRIMARY KEY,
            price          REAL NOT NULL,
            stock_quantity INTEGER NOT NULL DEFAULT 0,
            updated_at     TEXT
        )
    ''')
    conn.commit()
    conn.close()


def seed_inventory(items):
    """
    items: iterable of (item_name, price, stock_quantity) tuples.
    Inserts any item_name not already present. Never overwrites an
    existing row (so calling this again after the admin has adjusted
    stock — e.g. every startup() — doesn't silently reset it).
    """
    init_inventory_table()
    conn = _get_conn()
    cursor = conn.cursor()
    for name, price, qty in items:
        cursor.execute("SELECT 1 FROM inventory WHERE item_name = ?", (name,))
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO inventory (item_name, price, stock_quantity, updated_at) VALUES (?, ?, ?, ?)",
                (name, price, qty, time.strftime("%Y-%m-%d %H:%M:%S"))
            )
    conn.commit()
    conn.close()


def get_inventory():
    """Returns [{'item_name','price','stock_quantity','updated_at'}, ...] sorted by name."""
    init_inventory_table()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT item_name, price, stock_quantity, updated_at FROM inventory ORDER BY item_name")
    rows = cursor.fetchall()
    conn.close()
    return [
        {"item_name": r[0], "price": r[1], "stock_quantity": r[2], "updated_at": r[3]}
        for r in rows
    ]


def get_item(item_name):
    """Single item lookup (case-sensitive on the stored name). Returns dict or None."""
    init_inventory_table()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT item_name, price, stock_quantity, updated_at FROM inventory WHERE item_name = ?",
        (item_name,)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {"item_name": row[0], "price": row[1], "stock_quantity": row[2], "updated_at": row[3]}


def set_stock_quantity(item_name, quantity):
    """Set the absolute stock quantity for an item. Returns True if a row was updated."""
    init_inventory_table()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE inventory SET stock_quantity = ?, updated_at = ? WHERE item_name = ?",
        (quantity, time.strftime("%Y-%m-%d %H:%M:%S"), item_name)
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def add_stock(item_name, add_quantity):
    """
    Increments stock by add_quantity (pass a negative number to
    decrement — e.g. after a sale, if checkout is ever wired to call
    this). Returns True if a row was updated.
    """
    init_inventory_table()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE inventory SET stock_quantity = stock_quantity + ?, updated_at = ? WHERE item_name = ?",
        (add_quantity, time.strftime("%Y-%m-%d %H:%M:%S"), item_name)
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def set_price(item_name, price):
    """Update an item's price. Returns True if a row was updated."""
    init_inventory_table()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE inventory SET price = ?, updated_at = ? WHERE item_name = ?",
        (price, time.strftime("%Y-%m-%d %H:%M:%S"), item_name)
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated
