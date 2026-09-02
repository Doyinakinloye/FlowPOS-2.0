"""
SMART INVENTORY SYSTEM
Admin Panel — Management & Reporting Utilities
"""

import os
import numpy as np

from face_db import get_all_customers, delete_customer, customer_exists, save_customer, get_all_transactions
from embedding_loader import remove_embedding_from_memory, add_embedding_to_memory
from services.shopping_session import list_active_sessions
from services.inventory import get_inventory, set_stock_quantity, add_stock, set_price
from services.enrollment_manager import list_pending_duplicates, clear_pending_duplicate


def view_active_sessions():
    sessions = list_active_sessions()
    if not sessions:
        print("\n[Admin] No active shopping sessions (PINs) at the moment.")
        return

    print("\n" + "=" * 50)
    print("   ACTIVE SHOPPING SESSIONS (LIVE PINs)")
    print("=" * 50)
    for s in sessions:
        print(f"\nCustomer : {s['customer_name']} ({s['customer_id']})")
        print(f"PIN      : {s['pin']}")
        print(f"Started  : {s['created_at']}")
        print("-" * 35)


def view_customers():
    customers = get_all_customers()
    if not customers:
        print("\n[Admin] No customers enrolled yet!")
        return

    print("\n" + "=" * 50)
    print("   ENROLLED CUSTOMERS DATABASE")
    print("=" * 50)
    for c in customers:
        status  = "✅" if c.get("embedding") is not None else "❌"
        unknown = " [UNKNOWN]" if str(c["staff_id"]).startswith("UNK-") else ""
        print(f"\n{status} {c['name']}{unknown}")
        print(f"   ID       : {c['staff_id']}")
        print(f"   Enrolled : {c.get('enrolled_at', 'N/A')}")
        print(f"   Purchases: {c.get('total_purchases', 0)}")
        print(f"   Spent    : ₦{c.get('total_spent', 0.0):.2f}")
        print("-" * 35)


def delete_customer_menu():
    view_customers()
    staff_id = input("\nStaff ID to delete: ").strip()
    if not customer_exists(staff_id):
        print("[Admin] Customer ID not found!")
        return
    confirm = input(f"Delete customer {staff_id}? (y/n): ").strip().lower()
    if confirm == "y":
        folder = None
        for c in get_all_customers():
            if c["staff_id"] == staff_id:
                folder = c.get("folder_path")
                break

        if delete_customer(staff_id):
            remove_embedding_from_memory(staff_id)
            if folder:
                import os, shutil
                if os.path.isdir(folder):
                    shutil.rmtree(folder, ignore_errors=True)
            print("[Admin] Customer deleted successfully!")
    else:
        print("[Admin] Action cancelled.")


def view_inventory():
    items = get_inventory()
    if not items:
        print("\n[Admin] No inventory items on record yet!")
        return

    print("\n" + "=" * 50)
    print("   INVENTORY")
    print("=" * 50)
    print(f"{'ITEM':<16}{'QTY':<8}{'PRICE':<10}{'UPDATED':<20}")
    print("-" * 50)
    for it in items:
        print(f"{it['item_name']:<16}{it['stock_quantity']:<8}₦{it['price']:<9.0f}{it['updated_at'] or '':<20}")
    print("-" * 50)


def inventory_menu():
    while True:
        view_inventory()
        print("\n  1. Add Stock (restock an item)")
        print("  2. Set Exact Stock Quantity")
        print("  3. Update Price")
        print("  0. Back to Admin Menu")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            name = input("Item name: ").strip().lower()
            try:
                qty = int(input("Quantity to add: ").strip())
            except ValueError:
                print("[Admin] Invalid quantity — enter a whole number.")
                continue
            if add_stock(name, qty):
                print(f"[Admin] Added {qty} to '{name}'.")
            else:
                print(f"[Admin] '{name}' not found in inventory.")

        elif choice == "2":
            name = input("Item name: ").strip().lower()
            try:
                qty = int(input("New exact quantity: ").strip())
            except ValueError:
                print("[Admin] Invalid quantity — enter a whole number.")
                continue
            if set_stock_quantity(name, qty):
                print(f"[Admin] Set '{name}' stock to {qty}.")
            else:
                print(f"[Admin] '{name}' not found in inventory.")

        elif choice == "3":
            name = input("Item name: ").strip().lower()
            try:
                price = float(input("New price (₦): ").strip())
            except ValueError:
                print("[Admin] Invalid price — enter a number.")
                continue
            if set_price(name, price):
                print(f"[Admin] Set '{name}' price to ₦{price:.0f}.")
            else:
                print(f"[Admin] '{name}' not found in inventory.")

        elif choice == "0":
            break
        else:
            print("Invalid option. Please try again.")


def view_flagged_enrollments():
    records = list_pending_duplicates()
    if not records:
        print("\n[Admin] No enrollments currently flagged for review.")
        return []

    print("\n" + "=" * 50)
    print("   FLAGGED ENROLLMENTS")
    print("=" * 50)
    for i, r in enumerate(records, start=1):
        reason = f"Possible duplicate of {r.get('match_name')} ({r.get('match_id')}) — {r.get('similarity', 0):.0%} similarity"
        photo = r.get("folder_path") or "(no photo saved)"
        print(f"\n{i}. Candidate ID : {r['staff_id']}")
        print(f"   Reason       : {reason}")
        print(f"   Photo        : {photo}")
        print(f"   Flagged at   : {r.get('flagged_at')}")
        print("-" * 35)
    return records


def _open_photo(path):
    """Opens the flagged-enrollment photo in the OS default viewer. Windows-only
    (os.startfile) -- matches this project's deployment target; a terminal
    admin panel can't render an image inline any other way."""
    if not path or not os.path.isfile(path):
        print("[Admin] No photo file found for this entry.")
        return
    try:
        os.startfile(path)
    except AttributeError:
        print(f"[Admin] Can't auto-open on this OS — photo is at: {path}")
    except Exception as e:
        print(f"[Admin] Could not open photo: {e}")


def _approve_flagged(record):
    """Admin override: enroll the flagged candidate as a real customer anyway,
    using the embeddings already captured at enrollment time."""
    staff_id = record["staff_id"]
    name = record["name"]
    embeddings = [np.array(e) for e in record["embeddings"]]
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)
    add_embedding_to_memory(staff_id, name, avg_emb)
    save_customer(staff_id, name, "", avg_emb)


def flagged_enrollments_menu():
    while True:
        records = view_flagged_enrollments()
        if not records:
            break
        print("\n  1. View photo for an entry")
        print("  2. Approve an entry (enroll anyway)")
        print("  3. Delete an entry (reject)")
        print("  0. Back to Admin Menu")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            cid = input("Candidate ID to view: ").strip()
            match = next((r for r in records if r["staff_id"] == cid), None)
            if not match:
                print("[Admin] Candidate ID not found.")
                continue
            _open_photo(match.get("folder_path"))

        elif choice == "2":
            cid = input("Candidate ID to approve: ").strip()
            match = next((r for r in records if r["staff_id"] == cid), None)
            if not match:
                print("[Admin] Candidate ID not found.")
                continue
            _approve_flagged(match)
            clear_pending_duplicate(cid)
            print(f"[Admin] Approved. {cid} is now an enrolled customer.")

        elif choice == "3":
            cid = input("Candidate ID to delete: ").strip()
            match = next((r for r in records if r["staff_id"] == cid), None)
            if not match:
                print("[Admin] Candidate ID not found.")
                continue
            photo = match.get("folder_path")
            clear_pending_duplicate(cid)
            if photo and os.path.isfile(photo):
                try:
                    os.remove(photo)
                except OSError:
                    pass
            print(f"[Admin] Deleted flagged entry {cid}.")

        elif choice == "0":
            break
        else:
            print("Invalid option. Please try again.")


def view_transactions():
    txns = get_all_transactions()   # face_db.py's existing function -- see checkout.py's comment on why reused, not duplicated
    if not txns:
        print("\n[Admin] No transactions recorded yet.")
        return

    print("\n" + "=" * 50)
    print("   PAST TRANSACTIONS")
    print("=" * 50)
    for t in txns:
        print(f"\n{t['customer_name']} ({t['customer_id']})")
        print(f"   Time  : {t['timestamp']}  ({t['session_duration']}s)")
        for it in t['items']:
            print(f"     {it['qty']}x {it['name']} @ N{it['price']}")
        print(f"   TOTAL : N{t['total']:.0f}")
        print("-" * 35)


def run_admin_panel():
    while True:
        print("\n" + "=" * 50)
        print("   ADMINISTRATIVE CONTROL PANEL")
        print("=" * 50)
        print("  1. View Active Shoppers & Live PINs")
        print("  2. View All Enrolled Customers")
        print("  3. Delete Customer Entry")
        print("  4. Check Inventory")
        print("  5. Review Flagged Enrollment")
        print("  6. View Past Transactions")
        print()
        print("  0. Back to Main Menu")
        print("=" * 50)

        choice = input("Enter admin choice: ").strip()

        if choice == "1":
            view_active_sessions()
        elif choice == "2":
            view_customers()
        elif choice == "3":
            delete_customer_menu()
        elif choice == "4":
            inventory_menu()
        elif choice == "5":
            flagged_enrollments_menu()
        elif choice == "6":
            view_transactions()
        elif choice == "0":
            break
        else:
            print("Invalid option. Please try again.")