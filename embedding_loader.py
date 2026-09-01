# ============================================
# SMART INVENTORY SYSTEM
# Module: Embedding Loader
# Author: P1 - Oladele Jamiu Adeyemi
# Description: Loads ALL face embeddings + centroids
# Centroid-first matching with individual-embedding fallback
# ============================================

import numpy as np
from face_db import get_all_customers_multi, init_database, compute_centroid

# Global DB — { staff_id: { name, embeddings[], centroid } }
EMBEDDING_DB = {}

# Recognition threshold. Raised from the old 0.45 — with good
# enrollment quality (passive pipeline) and centroid matching,
# genuine matches comfortably clear ~0.60 while strangers don't.
DEFAULT_THRESHOLD = 0.60

# If the centroid score lands within this margin below threshold,
# it's worth double-checking against the individual embeddings
# before giving up — covers the case where one stored pose is a
# much better match than the averaged centroid.
FALLBACK_MARGIN = 0.08


def load_embeddings():
    """Load all embeddings + centroids from SQLite into memory"""
    global EMBEDDING_DB

    try:
        init_database()
        customers = get_all_customers_multi()

        if not customers:
            print("No customers enrolled yet!")
            EMBEDDING_DB.clear()
            return 0

        EMBEDDING_DB.clear()
        loaded = 0
        skipped = 0

        for customer in customers:
            staff_id   = customer["staff_id"]
            embeddings = customer["embeddings"]
            centroid   = customer.get("centroid")

            if embeddings:
                if centroid is None:
                    centroid = compute_centroid(embeddings)

                EMBEDDING_DB[staff_id] = {
                    "name":        customer["name"],
                    "staff_id":    staff_id,
                    "embeddings":  embeddings,
                    "centroid":    centroid,
                    "enrolled_at": customer["enrolled_at"]
                }
                loaded += 1
                print(f"  Loaded {len(embeddings)} embeddings + centroid "
                      f"for {customer['name']}")
            else:
                print(f"Warning: No embedding for "
                      f"{customer['name']} — skipping")
                skipped += 1

        print(f"\nEmbeddings loaded: {loaded}")
        print(f"Skipped: {skipped}")
        print(f"Total in memory: {len(EMBEDDING_DB)}")
        return loaded

    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return 0

def get_embedding_db():
    """Get embedding database — reload if empty"""
    global EMBEDDING_DB
    load_embeddings()
    return EMBEDDING_DB

def add_embedding_to_memory(staff_id, name, embeddings):
    """Add customer embeddings + computed centroid to memory"""
    global EMBEDDING_DB
    if isinstance(embeddings, np.ndarray):
        embeddings = [embeddings]
    EMBEDDING_DB[staff_id] = {
        "name":        name,
        "staff_id":    staff_id,
        "embeddings":  embeddings,
        "centroid":    compute_centroid(embeddings),
        "enrolled_at": None
    }
    print(f"Added {len(embeddings)} embeddings + centroid for {name}")

def remove_embedding_from_memory(staff_id):
    """Remove customer from memory"""
    global EMBEDDING_DB
    if staff_id in EMBEDDING_DB:
        name = EMBEDDING_DB[staff_id]["name"]
        del EMBEDDING_DB[staff_id]
        print(f"Removed {name} from memory")
        return True
    return False

def get_embedding_count():
    """Return number of customers in memory"""
    return len(EMBEDDING_DB)

def cosine_similarity(e1, e2):
    """Cosine similarity between two embeddings"""
    try:
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(e1, e2) / (n1 * n2))
    except:
        return 0.0


def _best_individual_match(live_norm):
    """
    Fallback comparison — checks the live embedding against every
    individually stored embedding for every customer (the old
    behaviour). Used only when centroid matching is inconclusive.

    Returns (best_match_dict_or_None, best_score).
    """
    best_match = None
    best_score = 0.0

    for staff_id, data in EMBEDDING_DB.items():
        stored_embeddings = data["embeddings"]
        if not stored_embeddings:
            continue

        scores = []
        for stored_emb in stored_embeddings:
            stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-10)
            scores.append(float(np.dot(live_norm, stored_norm)))

        max_score = max(scores)
        avg_score = float(np.mean(scores))
        final_score = (max_score * 0.7) + (avg_score * 0.3)

        if final_score > best_score:
            best_score = final_score
            best_match = {
                "name":       data["name"],
                "staff_id":   staff_id,
                "similarity": final_score,
                "max_sim":    max_score,
                "method":     "individual"
            }

    return best_match, best_score


def find_match(live_embedding, threshold=DEFAULT_THRESHOLD):
    """
    Find the best matching customer.

    Strategy:
      1. Compare the live embedding to each customer's CENTROID
         first — fast, and more stable against lighting/angle
         variation than any single stored frame.
      2. If the best centroid score clearly clears the threshold,
         confirm immediately.
      3. If the best centroid score is close but inconclusive
         (within FALLBACK_MARGIN below threshold), fall back to
         comparing against all individual stored embeddings —
         catches cases where one specific enrolled pose is a much
         closer match than the averaged centroid.
      4. Otherwise, no match.

    Returns match dict or None.
    """
    global EMBEDDING_DB

    if not EMBEDDING_DB:
        load_embeddings()

    if not EMBEDDING_DB:
        return None

    live_norm = live_embedding / (np.linalg.norm(live_embedding) + 1e-10)

    # ── Step 1: centroid-first pass ──────────────────────────
    best_centroid_match = None
    best_centroid_score  = 0.0

    for staff_id, data in EMBEDDING_DB.items():
        centroid = data.get("centroid")
        if centroid is None:
            continue
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        score = float(np.dot(live_norm, centroid_norm))

        print(f"  {data['name']}: centroid_sim={score:.3f}")

        if score > best_centroid_score:
            best_centroid_score = score
            best_centroid_match = {
                "name":       data["name"],
                "staff_id":   staff_id,
                "similarity": score,
                "max_sim":    score,
                "method":     "centroid"
            }

    print(f"  Best (centroid): "
          f"{best_centroid_match['name'] if best_centroid_match else 'None'}"
          f" = {best_centroid_score:.3f} (threshold: {threshold})")

    if best_centroid_match and best_centroid_score >= threshold:
        return best_centroid_match

    # ── Step 2: inconclusive — fall back to individual embeddings ──
    if best_centroid_score >= (threshold - FALLBACK_MARGIN):
        print("  Centroid score inconclusive — checking individual "
              "embeddings...")
        fallback_match, fallback_score = _best_individual_match(live_norm)

        if fallback_match:
            print(f"  Best (individual): {fallback_match['name']} "
                  f"= {fallback_score:.3f} (threshold: {threshold})")

        if fallback_match and fallback_score >= threshold:
            return fallback_match

    return None