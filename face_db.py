import sqlite3
import numpy as np
import json
import os
import time


# CONFIGURATION

DB_PATH = "inventory.db"
CUSTOMERS_DIR = "customers"   # matches services/enrollment_manager.py

# DATABASE SETUP


def init_database():
    """Initialize SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create customers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                staff_id TEXT UNIQUE NOT NULL,
                folder_path TEXT,
                embedding_json TEXT,
                embeddings_json TEXT,
                enrolled_at TEXT,
                total_purchases INTEGER DEFAULT 0,
                total_spent REAL DEFAULT 0.0
            )
        ''')

        # Add embeddings_json column if not exists
        try:
            cursor.execute(
                "ALTER TABLE customers ADD COLUMN embeddings_json TEXT"
            )
            print("Added embeddings_json column")
        except:
            pass  # Column already exists

        # Add centroid_json column if not exists
        # Centroid = normalized average of all stored embeddings.
        # Computed once at enrollment time, used as the first/fast
        # comparison point during recognition.
        try:
            cursor.execute(
                "ALTER TABLE customers ADD COLUMN centroid_json TEXT"
            )
            print("Added centroid_json column")
        except:
            pass  # Column already exists

        # Create transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                items_json TEXT,
                total REAL DEFAULT 0.0,
                session_duration REAL DEFAULT 0.0
            )
        ''')

        conn.commit()
        conn.close()
        print("Database initialized successfully!")
        return True

    except Exception as e:
        print(f"Database initialization error: {e}")
        return False


def compute_centroid(embeddings):
    """
    Compute the normalized centroid (average) of a list of
    embeddings. This sits in the "middle" of all the pose/lighting
    variation captured at enrollment, so it tends to be a more
    stable match target than any single stored frame.

    embeddings: list of numpy arrays (same dimensionality)
    Returns a normalized numpy array, or None if input is empty.
    """
    if not embeddings:
        return None
    stacked = np.stack(embeddings)
    centroid = np.mean(stacked, axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return centroid
    return centroid / norm


def save_customer_multi_embedding(name, staff_id,
                                   folder_path, embeddings):
    """
    Save customer with multiple embeddings.
    embeddings: list of numpy arrays

    Also computes and stores the centroid (normalized average) of
    all embeddings — used by recognition as the first, fast
    comparison point before falling back to individual embeddings.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Convert list of embeddings to JSON
        embeddings_list = [emb.tolist() for emb in embeddings]
        embeddings_json = json.dumps(embeddings_list)

        # Also save first embedding as main embedding
        main_embedding_json = json.dumps(embeddings[0].tolist())

        # Compute and serialize centroid
        centroid = compute_centroid(embeddings)
        centroid_json = json.dumps(centroid.tolist()) if centroid is not None else None

        cursor.execute("SELECT id FROM customers WHERE staff_id = ?",
                      (staff_id,))
        existing = cursor.fetchone()

        if existing:
            cursor.execute('''
                UPDATE customers
                SET name=?, folder_path=?,
                    embedding_json=?,
                    embeddings_json=?,
                    centroid_json=?,
                    enrolled_at=?
                WHERE staff_id=?
            ''', (name, folder_path,
                  main_embedding_json,
                  embeddings_json,
                  centroid_json,
                  time.strftime("%Y-%m-%d %H:%M:%S"),
                  staff_id))
        else:
            cursor.execute('''
                INSERT INTO customers
                (name, staff_id, folder_path,
                 embedding_json, embeddings_json, centroid_json, enrolled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, staff_id, folder_path,
                  main_embedding_json,
                  embeddings_json,
                  centroid_json,
                  time.strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
        conn.close()
        print(f"Saved {len(embeddings)} embeddings + centroid for {name}")
        return True

    except Exception as e:
        print(f"Error saving customer: {e}")
        return False

def get_all_customers_multi():
    """Get customers with multiple embeddings + centroid"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT name, staff_id, folder_path,
                   embedding_json, embeddings_json,
                   enrolled_at, total_purchases, total_spent,
                   centroid_json
            FROM customers
        ''')

        rows = cursor.fetchall()
        conn.close()

        customers = []
        for row in rows:
            # Load multiple embeddings if available
            embeddings = []
            if row[4]:  # embeddings_json
                try:
                    emb_list = json.loads(row[4])
                    embeddings = [np.array(e) for e in emb_list]
                except:
                    pass

            # Fallback to single embedding
            if not embeddings and row[3]:
                try:
                    embeddings = [np.array(json.loads(row[3]))]
                except:
                    pass

            # Load centroid if available, else compute on the fly
            # from whatever embeddings we have (covers customers
            # enrolled before centroid_json existed).
            centroid = None
            if row[8]:  # centroid_json
                try:
                    centroid = np.array(json.loads(row[8]))
                except:
                    centroid = None
            if centroid is None and embeddings:
                centroid = compute_centroid(embeddings)

            customers.append({
                "name": row[0],
                "staff_id": row[1],
                "folder_path": row[2],
                "embedding": np.array(json.loads(row[3])) if row[3] else None,
                "embeddings": embeddings,
                "centroid": centroid,
                "enrolled_at": row[5],
                "total_purchases": row[6],
                "total_spent": row[7]
            })

        return customers

    except Exception as e:
        print(f"Error getting customers: {e}")
        return []

# ============================================
# CUSTOMER OPERATIONS
# ============================================

def save_customer(name, staff_id, folder_path, embedding):
    """
    Save customer to SQLite database
    name: customer full name
    staff_id: unique staff ID
    folder_path: path to customer photos
    embedding: numpy array from ArcFace
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Convert numpy embedding to JSON string for storage
        embedding_json = json.dumps(embedding.tolist())

        # Check if customer already exists
        cursor.execute(
            "SELECT id FROM customers WHERE staff_id = ?",
            (staff_id,)
        )
        existing = cursor.fetchone()

        if existing:
            # Update existing customer
            cursor.execute('''
                UPDATE customers
                SET name = ?,
                    folder_path = ?,
                    embedding_json = ?,
                    enrolled_at = ?
                WHERE staff_id = ?
            ''', (
                name,
                folder_path,
                embedding_json,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                staff_id
            ))
            print(f"Customer {name} updated in database!")
        else:
            # Insert new customer
            cursor.execute('''
                INSERT INTO customers
                (name, staff_id, folder_path, embedding_json, enrolled_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                name,
                staff_id,
                folder_path,
                embedding_json,
                time.strftime("%Y-%m-%d %H:%M:%S")
            ))
            print(f"Customer {name} saved to database!")

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"Error saving customer: {e}")
        return False


# ============================================
# UNKNOWN CUSTOMER CREATION
# ============================================

def create_unknown_customer(face_crop):
    """
    Create a temporary customer record for an unrecognized person
    so a session can open and purchases can be logged against a
    real ID, even though we don't know who they are yet.

    face_crop: numpy array (BGR image) — the photo to save, normally
               the sharpest buffered crop from a confirmed unknown
               vote in face_engine.py

    No embedding is stored — by definition, we never matched this
    person to anyone. The record exists purely so M4 has a
    customer_id to log against, and so a manager can later review
    the photo in admin_panel.py and assign a real name (at which
    point proper enrollment with embeddings still needs to happen
    separately — assigning a name does not retroactively generate
    an embedding for this person).

    Returns the new staff_id (e.g. "UNK-1781912345") on success,
    or None on failure.
    """
    try:
        staff_id    = f"UNK-{int(time.time())}"
        folder_name = f"unknown_{int(time.time())}"
        folder_path = os.path.join(CUSTOMERS_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Save the photo as best_frame.jpg — this is the exact
        # filename admin_panel.py's show_unknown_photo() looks
        # for first, before falling back to other names.
        photo_path = os.path.join(folder_path, "best_frame.jpg")
        if face_crop is not None:
            import cv2
            cv2.imwrite(photo_path, face_crop)
        else:
            # No crop available (shouldn't normally happen) — still
            # create the record so the session can open, just
            # without a photo. admin_panel.py already handles a
            # missing photo gracefully.
            print("Warning: creating unknown customer with no photo")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO customers
            (name, staff_id, folder_path, embedding_json, enrolled_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            "Unknown",
            staff_id,
            folder_path,
            None,
            time.strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()

        print(f"Created unknown customer record: {staff_id}")
        return staff_id

    except Exception as e:
        print(f"Error creating unknown customer: {e}")
        return None


def rename_unknown_customer(old_staff_id, new_name, new_staff_id):
    """
    Assign a real name to a temporary UNK- customer record.

    This is an UPDATE, not a delete-and-recreate — the row keeps its
    id, folder_path, and (importantly) total_purchases/total_spent.
    An unknown customer may have already made purchases under their
    temp ID while the session was open; deleting and recreating the
    row would silently lose that history.

    No embedding is added here. Renaming someone does not teach the
    system to recognize their face — that still requires running
    them through proper enrollment separately. Until that happens,
    this person will be marked unknown again next time they show up.

    Returns True on success, False on failure (e.g. new_staff_id
    already taken, since staff_id is UNIQUE).
    """
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id FROM customers WHERE staff_id = ?",
            (old_staff_id,)
        )
        if cursor.fetchone() is None:
            print(f"No customer found with staff_id {old_staff_id}")
            conn.close()
            return False

        cursor.execute('''
            UPDATE customers
            SET name = ?, staff_id = ?
            WHERE staff_id = ?
        ''', (new_name, new_staff_id, old_staff_id))

        conn.commit()
        conn.close()

        print(f"✅ {old_staff_id} → {new_name} ({new_staff_id})")
        return True

    except sqlite3.IntegrityError:
        print(f"staff_id {new_staff_id} already in use")
        return False
    except Exception as e:
        print(f"Rename error: {e}")
        return False

def get_most_recent_customer():
    """
    Return the most recently INSERTED customer row (highest id),
    excluding temporary UNK- records.

    Used by entrance.py: run_enrollment() only returns True/False,
    not the identity it just created, and its tested return contract
    is deliberately left untouched here. This lookup lets entrance.py
    find out who was just enrolled so it can open their shopping
    session immediately, without a second face scan.

    Returns a customer dict (same shape as get_all_customers()) or
    None if nothing was found.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT name, staff_id, folder_path,
                   embedding_json, enrolled_at,
                   total_purchases, total_spent
            FROM customers
            WHERE staff_id NOT LIKE 'UNK-%'
            ORDER BY id DESC LIMIT 1
        ''')
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "name": row[0],
            "staff_id": row[1],
            "folder_path": row[2],
            "embedding": np.array(json.loads(row[3])) if row[3] else None,
            "enrolled_at": row[4],
            "total_purchases": row[5],
            "total_spent": row[6],
        }
    except Exception as e:
        print(f"Error getting most recent customer: {e}")
        return None


def find_customer_by_name(name):
    """
    Case-insensitive exact-name lookup among properly enrolled
    (non-UNK) customers. Returns the matching customer dict, or None.
    """
    target = (name or "").strip().lower()
    if not target:
        return None
    for c in get_all_customers():
        if c["staff_id"].startswith("UNK-"):
            continue
        if (c["name"] or "").strip().lower() == target:
            return c
    return None 

def get_all_customers():
    """
    Get all customers from database
    Returns list of customer dictionaries
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT name, staff_id, folder_path,
                   embedding_json, enrolled_at,
                   total_purchases, total_spent
            FROM customers
        ''')

        rows = cursor.fetchall()
        conn.close()

        customers = []
        for row in rows:
            customers.append({
                "name": row[0],
                "staff_id": row[1],
                "folder_path": row[2],
                "embedding": np.array(json.loads(row[3])) if row[3] else None,
                "enrolled_at": row[4],
                "total_purchases": row[5],
                "total_spent": row[6]
            })

        return customers

    except Exception as e:
        print(f"Error getting customers: {e}")
        return []

def get_customer_by_id(staff_id):
    """
    Get single customer by staff ID
    Returns customer dictionary or None
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT name, staff_id, folder_path,
                   embedding_json, enrolled_at,
                   total_purchases, total_spent
            FROM customers WHERE staff_id = ?
        ''', (staff_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "name": row[0],
                "staff_id": row[1],
                "folder_path": row[2],
                "embedding": np.array(json.loads(row[3])) if row[3] else None,
                "enrolled_at": row[4],
                "total_purchases": row[5],
                "total_spent": row[6]
            }
        return None

    except Exception as e:
        print(f"Error getting customer: {e}")
        return None

def delete_customer(staff_id):
    """
    Delete customer from database
    Returns True if successful
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM customers WHERE staff_id = ?",
            (staff_id,)
        )

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        if affected > 0:
            print(f"Customer {staff_id} deleted from database!")
            return True
        else:
            print(f"Customer {staff_id} not found!")
            return False

    except Exception as e:
        print(f"Error deleting customer: {e}")
        return False

def customer_exists(staff_id):
    """
    Check if customer exists in database
    Returns True if exists
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id FROM customers WHERE staff_id = ?",
            (staff_id,)
        )

        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    except Exception as e:
        print(f"Error checking customer: {e}")
        return False

def update_customer_stats(staff_id, amount):
    """
    Update customer purchase statistics
    Called after each transaction
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE customers
            SET total_purchases = total_purchases + 1,
                total_spent = total_spent + ?
            WHERE staff_id = ?
        ''', (amount, staff_id))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"Error updating stats: {e}")
        return False


# ============================================
# TRANSACTION OPERATIONS
# ============================================

def save_transaction(customer_id, customer_name,
                     items, total, session_duration):
    """
    Save transaction to database
    items: dictionary of {item_name: quantity}
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO transactions
            (customer_id, customer_name, timestamp,
             items_json, total, session_duration)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            customer_id,
            customer_name,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            json.dumps(items),
            total,
            session_duration
        ))

        conn.commit()
        conn.close()
        print(f"Transaction saved for {customer_name}!")
        return True

    except Exception as e:
        print(f"Error saving transaction: {e}")
        return False

def get_all_transactions():
    """Get all transactions from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT customer_id, customer_name,
                   timestamp, items_json,
                   total, session_duration
            FROM transactions
            ORDER BY timestamp DESC
        ''')

        rows = cursor.fetchall()
        conn.close()

        transactions = []
        for row in rows:
            transactions.append({
                "customer_id": row[0],
                "customer_name": row[1],
                "timestamp": row[2],
                "items": json.loads(row[3]),
                "total": row[4],
                "session_duration": row[5]
            })

        return transactions

    except Exception as e:
        print(f"Error getting transactions: {e}")
        return []


# ============================================
# DATABASE INFO
# ============================================

def get_database_info():
    """Print database statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM customers")
        customer_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM transactions")
        transaction_count = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(total) FROM transactions")
        total_sales = cursor.fetchone()[0] or 0

        conn.close()

        print("\n" + "=" * 50)
        print("   DATABASE INFORMATION")
        print("=" * 50)
        print(f"Database file: {DB_PATH}")
        print(f"Total customers: {customer_count}")
        print(f"Total transactions: {transaction_count}")
        print(f"Total sales: ₦{total_sales:.2f}")
        print("=" * 50)

    except Exception as e:
        print(f"Error getting database info: {e}")


# ============================================
# TEST DATABASE
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("   SMART INVENTORY - DATABASE TEST")
    print("=" * 50)

    # Initialize database
    init_database()

    # Show database info
    get_database_info()

    print("\nDatabase module ready!")
    print("Tables created: customers, transactions")