import json
import re
from pathlib import Path


ORDERS_FILE = Path("data/orders.json")


def normalize_order_id(order_id: str | None) -> str | None:
    """
    Normalize harmless order-ID variations into the canonical
    dataset format: ORD-<digits>.
    """

    if order_id is None:
        return None

    value = str(order_id).strip().upper()

    if not value:
        return None

    # ORD-1001
    match = re.fullmatch(r"ORD-(\d+)", value)

    if match:
        return f"ORD-{match.group(1)}"

    # ORD1001
    match = re.fullmatch(r"ORD(\d+)", value)

    if match:
        return f"ORD-{match.group(1)}"

    # 1001
    match = re.fullmatch(r"\d+", value)

    if match:
        return f"ORD-{match.group(0)}"

    return None


def load_order_data() -> dict:
    with ORDERS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_orders() -> list[dict]:
    return load_order_data()["orders"]


def lookup_order(order_id: str) -> dict | None:

    order_data = load_order_data()

    orders = order_data["orders"]
    snapshot_at = order_data["snapshot_at"]

    normalized_id = normalize_order_id(order_id)

    # Safely reject malformed IDs.
    if normalized_id is None:
        return None

    for order in orders:

        if order["order_id"].upper() != normalized_id:
            continue

        return {
            "order_id": order["order_id"],
            "status": order["status"],
            "membership_tier": order["membership_tier"],
            "placed_at": order["placed_at"],
            "status_updated_at": order["status_updated_at"],
            "shipped_at": order["shipped_at"],
            "delivered_at": order["delivered_at"],
            "carrier": order["carrier"],
            "tracking_number": order["tracking_number"],
            "estimated_delivery": order["estimated_delivery"],
            "customer_safe_message": order["customer_safe_message"],
            "snapshot_at": snapshot_at,

            # The dataset contains no order total/amount.
            "amount_available": False,

            "items": [
                {
                    "name": item["name"],
                    "quantity": item["quantity"],
                    "final_sale": item["final_sale"],
                }
                for item in order["items"]
            ],
        }

    return None