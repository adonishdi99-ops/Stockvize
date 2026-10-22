"""
db.py — persistent storage for the vendor agent.

Tables:
- inventory(item TEXT PRIMARY KEY, stock REAL, unit TEXT)
- orders(id INTEGER PRIMARY KEY, customer TEXT, items_json TEXT,
         location TEXT, status TEXT, created_at TEXT)
- processed_messages(message_id TEXT PRIMARY KEY, processed_at TEXT)
    -- tracks WhatsApp's own message IDs so a redelivered webhook
    -- event (which WhatsApp does send sometimes) never creates a
    -- duplicate order.

Run this file directly once to create/reset the DB with starter stock:
    python3 db.py
"""

import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = "vendor_agent.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            item TEXT PRIMARY KEY,
            stock REAL NOT NULL,
            unit TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer TEXT,
            items_json TEXT NOT NULL,
            location TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_messages (
            message_id TEXT PRIMARY KEY,
            processed_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_recent_history(sender_id: str, limit: int = 6):
    """Returns the last `limit` turns for this sender, oldest first, as
    [{"role": "user"/"assistant", "content": str}, ...] ready to drop into
    a chat completions messages list."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content FROM conversation_history WHERE sender_id = ? ORDER BY id DESC LIMIT ?",
        (sender_id, limit),
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_conversation_turn(sender_id: str, role: str, content: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO conversation_history (sender_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (sender_id, role, content, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def seed_starter_stock():
    """Only inserts items that don't already exist — safe to call every run."""
    conn = get_conn()
    starter = [("rice", 50, "bags"), ("garri", 30, "bags"), ("beans", 20, "cups")]
    for item, stock, unit in starter:
        conn.execute(
            "INSERT OR IGNORE INTO inventory (item, stock, unit) VALUES (?, ?, ?)",
            (item, stock, unit),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Message deduplication
# ---------------------------------------------------------------------------

def is_message_processed(message_id: str) -> bool:
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM processed_messages WHERE message_id = ?", (message_id,)).fetchone()
    conn.close()
    return row is not None


def mark_message_processed(message_id: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO processed_messages (message_id, processed_at) VALUES (?, ?)",
        (message_id, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Inventory operations
# ---------------------------------------------------------------------------

def db_get_stock(item: str):
    conn = get_conn()
    row = conn.execute("SELECT stock, unit FROM inventory WHERE item = ?", (item.lower(),)).fetchone()
    conn.close()
    if row is None:
        return None
    return row["stock"], row["unit"]


def db_update_stock(item: str, quantity_change: float, unit: str = ""):
    """Adds quantity_change to current stock (negative to subtract). Creates the item if new."""
    conn = get_conn()
    row = conn.execute("SELECT stock FROM inventory WHERE item = ?", (item.lower(),)).fetchone()
    if row is None:
        new_stock = max(quantity_change, 0)
        conn.execute(
            "INSERT INTO inventory (item, stock, unit) VALUES (?, ?, ?)",
            (item.lower(), new_stock, unit),
        )
    else:
        new_stock = row["stock"] + quantity_change
        conn.execute("UPDATE inventory SET stock = ? WHERE item = ?", (new_stock, item.lower()))
    conn.commit()
    conn.close()
    return new_stock


# ---------------------------------------------------------------------------
# Order operations
# ---------------------------------------------------------------------------

def db_create_order(customer: str, items: list, location: str = ""):
    """items: list of {"item": str, "quantity": number, "unit": str}. Returns (order_id, warnings)."""
    conn = get_conn()
    warnings = []

    for entry in items:
        item = entry["item"].lower()
        qty = entry["quantity"]
        row = conn.execute("SELECT stock FROM inventory WHERE item = ?", (item,)).fetchone()
        current = row["stock"] if row else 0
        if current < qty:
            warnings.append(f"{item}: requested {qty}, only {current} in stock")

    cur = conn.execute(
        "INSERT INTO orders (customer, items_json, location, status, created_at) VALUES (?, ?, ?, 'pending', ?)",
        (customer, json.dumps(items), location, datetime.now(timezone.utc).isoformat()),
    )
    order_id = cur.lastrowid

    for entry in items:
        item = entry["item"].lower()
        qty = entry["quantity"]
        row = conn.execute("SELECT stock FROM inventory WHERE item = ?", (item,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO inventory (item, stock, unit) VALUES (?, ?, '')", (item, -qty))
        else:
            conn.execute("UPDATE inventory SET stock = stock - ? WHERE item = ?", (qty, item))

    conn.commit()
    conn.close()
    return order_id, warnings


def db_update_order_location(order_id: str, location: str):
    conn = get_conn()
    row = conn.execute("SELECT id FROM orders WHERE id = ?", (order_id,)).fetchone()
    if row is None:
        conn.close()
        return False
    conn.execute("UPDATE orders SET location = ? WHERE id = ?", (location, order_id))
    conn.commit()
    conn.close()
    return True


def db_mark_order_status(order_id: str, status: str):
    conn = get_conn()
    row = conn.execute("SELECT id FROM orders WHERE id = ?", (order_id,)).fetchone()
    if row is None:
        conn.close()
        return False
    conn.execute("UPDATE orders SET status = ? WHERE id = ?", (status, order_id))
    conn.commit()
    conn.close()
    return True


def db_get_order(order_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def db_most_recent_order():
    conn = get_conn()
    row = conn.execute("SELECT * FROM orders ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


if __name__ == "__main__":
    init_db()
    seed_starter_stock()
    print(f"Database ready at {DB_PATH}")
    conn = get_conn()
    print("\nInventory:")
    for row in conn.execute("SELECT * FROM inventory"):
        print(f"  {row['item']}: {row['stock']} {row['unit']}")
    conn.close()