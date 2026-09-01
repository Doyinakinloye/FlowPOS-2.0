# billing.py
# Updated to work with Member 3's ItemEngine tripwire events.
# 'taken' = add to bill, 'returned' = subtract from bill

import datetime
from m4.price_catalog import get_price

class LiveSession:
    def __init__(self, customer_name, staff_id):
        self.customer_name  = customer_name
        self.staff_id       = staff_id
        self.start_time     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.running_total  = 0
        self.billed_items   = []

    def process_event(self, event: dict):
        """
        Call this with every event from ItemEngine.detect_with_tripwire().
        event = {'track_id': 3, 'label': 'cheese_balls',
                 'direction': 'taken', 'price': 200}

        'taken'    → adds to the bill
        'returned' → subtracts from the bill
        """
        label     = event["label"]
        direction = event["direction"]
        price     = event["price"]   # already signed: positive=taken, negative=returned

        record = {
            "customer_name": self.customer_name,
            "staff_id":      self.staff_id,
            "timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "item":          label,
            "direction":     direction,
            "unit_price":    abs(price),
            "line_total":    price,
            "running_total": self.running_total + price
        }

        self.running_total += price
        self.billed_items.append(record)

        action = "BILLED" if direction == "taken" else "RETURNED"
        print(f"{action}: {label} — N{abs(price)} | Running total: N{self.running_total}")

        return record

    def bill_item(self, item_label: str):
        """
        Fallback: manually bill a single item (for testing without the camera).
        """
        unit_price = get_price(item_label)
        event = {
            "label":     item_label,
            "direction": "taken",
            "price":     unit_price
        }
        return self.process_event(event)

    def bill_multiple_items(self, item_labels: list):
        bills = []
        for label in item_labels:
            bills.append(self.bill_item(label))
        return bills

    def get_session_summary(self):
        return {
            "customer_name": self.customer_name,
            "staff_id":      self.staff_id,
            "start_time":    self.start_time,
            "end_time":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "billed_items":  self.billed_items,
            "total":         self.running_total
        }

    def reset(self):
        self.running_total = 0
        self.billed_items  = []