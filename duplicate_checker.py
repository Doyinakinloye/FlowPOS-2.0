"""
SMART INVENTORY SYSTEM
Module: Duplicate Checker

Two entry points:
  check_duplicate(image_path)         — file-path based (used by legacy modes)
  check_duplicate_from_embedding(emb) — pre-generated embedding (smart enrollment)
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_db import get_all_customers_multi
from embedding_loader import cosine_similarity

# ── CONFIG ────────────────────────────────────────────────────
# Compare new embedding against ALL stored embeddings per customer.
# A customer is flagged as duplicate when ANY stored embedding
# exceeds this threshold.
#
# This value is imported from enrollment.py rather than defined
# here, so there's exactly one number to tune, not two. enrollment.py
# is the one place that's actually been tuned against real test data
# (0.75) — this module used to carry its own separate 0.88, which
# was silently ignored by every caller (they only read the raw
# similarity score back and re-judged it themselves), so it never
# actually did anything. Importing avoids that trap happening again.
try:
    from enrollment.enrollment import DUPLICATE_THRESHOLD
except ImportError:
    # Fallback if enrollment.py isn't importable in this context
    # (e.g. standalone testing) — keep in sync with enrollment.py
    DUPLICATE_THRESHOLD = 0.75


# ── CORE COMPARISON ───────────────────────────────────────────

def _compare_embedding_against_db(new_embedding):
    """
    Compare new_embedding (numpy array) against every stored
    embedding for every customer.

    Strategy: for each customer take the MAX similarity score
    across all their stored embeddings — this is the most
    permissive (safest) check for duplicates.

    Returns (is_duplicate, best_name, best_id, best_similarity).
    """
    customers = get_all_customers_multi()
    if not customers:
        return False, None, None, 0.0

    best_sim  = 0.0
    best_name = None
    best_id   = None

    for c in customers:
        stored_embeddings = c.get("embeddings", [])

        # Fallback: single embedding field
        if not stored_embeddings and c.get("embedding") is not None:
            stored_embeddings = [c["embedding"]]

        if not stored_embeddings:
            continue

        # Take the highest similarity across all stored embeddings
        sims = [cosine_similarity(new_embedding, e) for e in stored_embeddings]
        max_sim = max(sims)

        if max_sim > best_sim:
            best_sim  = max_sim
            best_name = c["name"]
            best_id   = c["staff_id"]

    if best_sim >= DUPLICATE_THRESHOLD:
        print(f"⚠️  Duplicate detected — {best_name} ({best_id})  "
              f"similarity={best_sim:.3f}")
        return True, best_name, best_id, best_sim

    print(f"✅ No duplicate  (max similarity={best_sim:.3f})")
    return False, None, None, best_sim


# ── PUBLIC API ────────────────────────────────────────────────

def check_duplicate_from_embedding(new_embedding):
    """
    Check pre-generated embedding against DB.
    Called by smart_enrollment after ArcFace step.

    Returns (is_duplicate, match_name, match_id, similarity).
    """
    try:
        return _compare_embedding_against_db(new_embedding)
    except Exception as e:
        print(f"Duplicate check error: {e}")
        return False, None, None, 0.0



# ── SELF-TEST ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   DUPLICATE CHECKER — ready")
    print(f"   Threshold : {DUPLICATE_THRESHOLD}")
    print("=" * 50)