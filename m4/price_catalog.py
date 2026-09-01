# price_catalog.py
# This file contains all drink prices in Naira.
#
# UPDATED 2026-08-31: switched to the new 4-class model (coke, fanta,
# sprite, water) that replaced the old 8-item snack catalog. The old
# prices are gone — best.pt no longer detects those items, so keeping
# stale entries around would just be confusing.

import datetime

PRICES = {
    "coke":   500,
    "fanta":  500,
    "sprite": 500,
    "water":  300,
}

def get_price(item_label):
    """
    Give this function an item name and it returns the price.
    If the item is not in the list it returns 0, warns you,
    and logs the unknown item to a file.
    """
    price = PRICES.get(item_label, 0)
    if price == 0:
        print(f"WARNING: '{item_label}' not in price catalog. Price set to 0.")
        with open("unknown_items_log.txt", "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] Unknown item detected: '{item_label}'\n")
    return price
