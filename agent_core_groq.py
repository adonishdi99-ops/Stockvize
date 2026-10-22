"""
agent_core_groq.py — the message-parsing brain for the vendor agent.

Turns messy WhatsApp-style text (typed OR transcribed from voice) into
structured actions via Groq's tool-calling, backed by a real SQLite DB.

Requires GROQ_API_KEY in .env.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

from db import (
    init_db, seed_starter_stock,
    db_get_stock, db_update_stock, db_create_order,
    db_mark_order_status, db_get_order, db_most_recent_order,
    db_update_order_location,
    get_recent_history, save_conversation_turn,
)

load_dotenv()
client = Groq()

MODEL = "openai/gpt-oss-120b"  # bigger, stronger reasoning than the deprecated llama-3.3-70b, still free tier

init_db()
seed_starter_stock()

# ---------------------------------------------------------------------------
# 1. TOOL SCHEMA
# ---------------------------------------------------------------------------

def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOLS = [
    tool(
        "create_order",
        "Record a new customer order. Use when a CUSTOMER message clearly requests items to buy/deliver.",
        {
            "customer_name": {"type": "string", "description": "Name if known, else 'unknown'"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit": {"type": "string"},
                    },
                    "required": ["item", "quantity"],
                },
            },
            "delivery_location": {"type": "string", "description": "Empty string if pickup or not mentioned"},
        },
        ["items"],
    ),
    tool(
        "update_inventory",
        "Adjust stock levels. Use when the VENDOR reports restocking, selling, or correcting inventory.",
        {
            "item": {"type": "string"},
            "quantity_change": {"type": "number", "description": "Positive to add stock, negative to remove"},
            "unit": {"type": "string"},
        },
        ["item", "quantity_change"],
    ),
    tool(
        "get_stock_level",
        "Look up current stock for an item.",
        {"item": {"type": "string"}},
        ["item"],
    ),
    tool(
        "mark_order_status",
        "Update an existing order's status.",
        {
            "order_id": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "delivered", "cancelled"]},
        },
        ["order_id", "status"],
    ),
    tool(
        "update_order_location",
        "Correct/update the delivery location on an already-created order. Use when a customer follows up to fix or clarify a location on a recent order (e.g. after being asked to confirm), rather than treating it as a brand new order.",
        {
            "order_id": {"type": "string"},
            "location": {"type": "string"},
        },
        ["order_id", "location"],
    ),
    tool(
        "get_last_order",
        "Look up the most recently placed order. Use when someone references 'that order', 'the last order', 'it' with no order number given.",
        {},
        [],
    ),
    tool(
        "reply_customer",
        "Send a plain reply with no data change — greetings, FAQs, unclear messages.",
        {"message": {"type": "string"}},
        ["message"],
    ),
]

KNOWN_ITEMS = ["rice", "garri", "beans"]  # replace with the real shop's stock list later

SYSTEM_PROMPT = f"""You are the message-parsing brain for a small shop's WhatsApp agent.
You will receive one message at a time, labeled as coming from either a CUSTOMER
or the VENDOR. Decide what real-world action it maps to and call exactly one tool.

The shop's ONLY known items are: {", ".join(KNOWN_ITEMS)}. You have no knowledge of
anything else this shop does or does not sell, its prices, hours, or policies beyond
what these tools tell you.

CRITICAL — never fabricate facts about the shop. This rule applies ONLY to
FACTUAL CLAIMS the customer asks about — does the shop sell X, what are your
hours, do you deliver to Y, etc. — things no tool can answer. It does NOT apply
to placing an order: if the item matches (or is a plausible typo of) a
KNOWN_ITEMS entry and a quantity is given, that is a real order and you MUST
call create_order immediately — do not defer it with "I'll check and get back
to you." That phrase is reserved strictly for questions about things the tools
cannot look up, never for order requests you can already fulfill.

ITEM NORMALIZATION: customer messages will have typos and slang (and some
messages are auto-transcribed from voice notes, which introduces its own
spelling/phonetic errors — e.g. "rise"/"rize"/"rice" or "gari"/"garri"). Map
misspellings to the closest KNOWN_ITEMS name. If an item clearly isn't in
KNOWN_ITEMS and isn't a plausible typo/mishearing of one, treat the message as
needing clarification (reply_customer) rather than inventing a new item name.

You may see earlier turns from this same sender above the current message —
use them to understand follow-ups and corrections (e.g. a bare location sent
right after an order, or "I said X" style corrections) instead of treating
every message as a fresh, unrelated request. Conversation history tells you
INTENT and CONTEXT; for exact IDs and current data (e.g. which order is
"the last one"), still call get_last_order rather than guessing from memory.

VOICE TRANSCRIPTION CAUTION: if a message is marked [TRANSCRIBED VOICE NOTE],
treat place names, proper nouns, and anything that doesn't sound like a real
Nigerian location/name as SUSPECT — transcription errors are common there
specifically. If a delivery location looks garbled, nonsensical, or like it
could be a mishearing (e.g. "EvoRoot", "kombodi ja"), do NOT guess what it
was meant to be and do NOT silently default to pickup — call create_order
with delivery_location left empty AND follow up with reply_customer asking
them to confirm/retype the delivery location. Quantities that seem to have
lost a word (e.g. "bag the cells" where a number seems to be missing) should
also prompt a clarifying reply_customer rather than a confident guess.

IMPORTANT: reply_customer does NOT save anything anywhere. It is ONLY for messages
with no data to record — greetings, vague questions, chit-chat, or things you
genuinely don't know. If the message contains a concrete item + quantity for an
order, or a concrete stock question, you MUST call the matching data tool
(create_order / update_inventory / get_stock_level / mark_order_status), never
reply_customer, even if you also want to sound friendly.

EXAMPLES:
Message: [CUSTOMER]: please send two bags of rice to my house
Correct action: call create_order with items=[{{"item": "rice", "quantity": 2, "unit": "bags"}}], delivery_location="customer's house". Do NOT reply_customer to "confirm" first — just create it.

Message: [CUSTOMER]: abeg i wan buy 3 bag rise n 2 beans deliver ikeja
Correct action: call create_order with items=[{{"item": "rice", "quantity": 3, "unit": "bag"}}, {{"item": "beans", "quantity": 2, "unit": ""}}], delivery_location="ikeja". "rise" is a typo of "rice" — normalize it silently, do not ask the customer to confirm the spelling.

Message: [CUSTOMER]: I want some rice
Correct action: call reply_customer asking for quantity — no number was given, so this is genuinely incomplete, unlike the examples above.

Message: [VENDOR/CUSTOMER context]: you just asked the customer to confirm a delivery location on their recent order, and they now reply with just the corrected location (e.g. "I said Ikeja", "it's Yaba not Yola")
Correct action: call get_last_order to find that order's ID, then call update_order_location with that ID and the corrected location. Do NOT ask them to restate their whole order from scratch — the order already exists, it just needs its location fixed.

Trigger rules (follow these exactly):
- Customer names specific item(s) + quantity(ies) they want -> ALWAYS create_order immediately. customer_name and delivery_location are OPTIONAL — never withhold order creation just because a name or location is missing; use "unknown" / "" for those fields instead.
- Vendor says to add/restock/remove/sell stock with a clear NUMBER -> ALWAYS update_inventory with that number as the change.
- Vendor gives an absolute target instead of a delta (e.g. "make it 0", "set stock to 5") -> call get_stock_level FIRST to see the current amount, then call update_inventory with the calculated delta needed to reach that target. Do not guess the delta blindly.
- Anyone asks "how much X do we have / is left / in stock" -> ALWAYS get_stock_level. Never say "let me check" in text — actually call the tool.
- Vendor references an order by an EXACT number ("order 12") + a status word -> ALWAYS mark_order_status with that number as order_id. If they reference an order vaguely ("that order", "the last one", "it") with NO digit given, call get_last_order FIRST to find its ID, then call mark_order_status with that ID and the requested status — do this as two steps, don't stop after just looking it up.
- A short follow-up message that looks like a location correction (a place name, "I said X", "it's X not Y") right after you or the customer discussed a delivery address -> call get_last_order to find the order, then update_order_location. Never treat this as a request to start a new order.
- If the previous turn in the conversation history was an order you just created, and the customer's next message is just a bare place name or "deliver to X" with no items mentioned -> that is delivery info for the order you just made, not a new order. Call get_last_order to confirm the ID, then update_order_location.
- Only use reply_customer for: greetings, unclear/incomplete requests, questions with no data lookup involved, or anything not covered by the known items/tools.
- Never guess a quantity that wasn't stated — if quantity is truly missing, use reply_customer to ask for it.
- You may call more than one tool in sequence when a task genuinely needs it (e.g. look up then act). Don't call tools you don't need. Once the necessary action(s) are done, you may stop — you don't need to call reply_customer just to "confirm" after already taking a data action.
"""


# ---------------------------------------------------------------------------
# 2. DISPATCH — real DB-backed functions. Each returns {"log": ..., "customer_message": ...}
#    log = internal/technical detail (terminal only)
#    customer_message = clean text safe to send over WhatsApp, or None if this
#    tool is an internal/vendor-only action or intermediate lookup step.
# ---------------------------------------------------------------------------

def create_order(customer_name="unknown", items=None, delivery_location=""):
    order_id, warnings = db_create_order(customer_name, items or [], delivery_location)
    log = f"Order #{order_id} created for {customer_name}: {items} -> {delivery_location or 'pickup'}"
    item_summary = ", ".join(
        f"{i.get('quantity')} {i.get('unit', '').strip()} {i.get('item')}".replace("  ", " ").strip()
        for i in (items or [])
    )
    customer_message = f"Your order (#{order_id}) for {item_summary} has been placed!"
    if warnings:
        log += f" | STOCK WARNING: {'; '.join(warnings)}"
        customer_message += " Note: some items are low/out of stock — the vendor will confirm availability."
    return {"log": log, "customer_message": customer_message}


def update_inventory(item, quantity_change, unit=""):
    new_stock = db_update_stock(item, quantity_change, unit)
    log = f"{item} stock now {new_stock} {unit}".strip()
    return {"log": log, "customer_message": None}


def get_stock_level(item):
    result = db_get_stock(item)
    if result is None:
        log = f"{item}: not tracked yet (0 in stock)"
    else:
        stock, unit = result
        log = f"{item}: {stock} {unit} in stock".strip()
    return {"log": log, "customer_message": None}


def update_order_location(order_id, location):
    ok = db_update_order_location(order_id, location)
    log = f"Order #{order_id} location updated to {location}" if ok else f"Order #{order_id} not found"
    customer_message = f"Got it — order #{order_id} will now go to {location}." if ok else None
    return {"log": log, "customer_message": customer_message}


def mark_order_status(order_id, status):
    ok = db_mark_order_status(order_id, status)
    log = f"Order #{order_id} marked {status}" if ok else f"Order #{order_id} not found"
    customer_message = f"Order #{order_id} is now marked as {status}." if ok else None
    return {"log": log, "customer_message": customer_message}


def get_last_order(dummy=None):
    order = db_most_recent_order()
    if order is None:
        log = "No orders yet."
    else:
        log = f"Most recent order: #{order['id']}, status={order['status']}, items={order['items_json']}"
    return {"log": log, "customer_message": None}


def reply_customer(message):
    return {"log": message, "customer_message": message}


DISPATCH = {
    "create_order": create_order,
    "update_inventory": update_inventory,
    "get_stock_level": get_stock_level,
    "mark_order_status": mark_order_status,
    "update_order_location": update_order_location,
    "get_last_order": get_last_order,
    "reply_customer": reply_customer,
}


# ---------------------------------------------------------------------------
# 3. PIPELINE
# ---------------------------------------------------------------------------

def handle_message(sender_role: str, text: str, sender_id: str = None, is_voice: bool = False) -> dict:
    """sender_id: the real WhatsApp number of whoever sent the message, if known
    (used to auto-fill customer_name on orders instead of relying on the model
    to guess an identity from message text, which it can't do).

    is_voice: True if this text came from transcribing a voice note — flags
    it to the model so it treats place names/quantities more cautiously.

    Returns {"customer_reply": str, "debug_log": str} — customer_reply is the
    ONLY thing that should ever be sent back over WhatsApp; debug_log is for
    your own terminal/logs and may contain internal formatting not meant for
    customers to see.
    """
    label = "[TRANSCRIBED VOICE NOTE]" if is_voice else ""
    current_turn_text = f"[{sender_role.upper()}]{label}: {text}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if sender_id:
        # prior turns for this exact sender, so a follow-up like "deliver it
        # to Bodija" or "I said Ikeja" has something to attach to — without
        # this, every message starts a brand-new conversation with zero
        # memory of anything said before, even seconds earlier.
        messages.extend(get_recent_history(sender_id))
    messages.append({"role": "user", "content": current_turn_text})

    debug_log = []
    customer_messages = []
    max_steps = 4

    for step in range(max_steps):
        forced = step == 0
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=1000,
                messages=messages,
                tools=TOOLS,
                tool_choice="required" if forced else "auto",
            )
        except Exception as e:
            if forced:
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        max_tokens=1000,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                except Exception:
                    debug_log.append(f"[error] Groq request failed twice: {e}")
                    break
            else:
                debug_log.append(f"[error] Groq request failed: {e}")
                break
        msg = response.choices[0].message

        if not msg.tool_calls:
            if msg.content and not customer_messages:
                customer_messages.append(msg.content)
                debug_log.append(f"[final text] {msg.content}")
            break

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        stop_now = False
        for call in msg.tool_calls:
            fn = DISPATCH.get(call.function.name)
            args = json.loads(call.function.arguments)
            if call.function.name == "create_order" and sender_id:
                args["customer_name"] = sender_id

            if not fn:
                result = {"log": f"Unknown tool: {call.function.name}", "customer_message": None}
            else:
                result = fn(**args)

            debug_log.append(f"[tool: {call.function.name}] {result['log']}")
            if result.get("customer_message"):
                customer_messages.append(result["customer_message"])

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result["log"]),
            })
            if call.function.name == "reply_customer":
                stop_now = True

        if stop_now:
            break

    customer_reply = "\n".join(customer_messages) if customer_messages else "Got it, thanks!"

    if sender_id:
        save_conversation_turn(sender_id, "user", current_turn_text)
        save_conversation_turn(sender_id, "assistant", customer_reply)

    return {
        "customer_reply": customer_reply,
        "debug_log": "\n".join(debug_log) if debug_log else "(no action taken)",
    }


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("No GROQ_API_KEY found. Create a .env file in this folder with:")
        print("  GROQ_API_KEY=your_key_here")
        print("Get a free key at https://console.groq.com")
        raise SystemExit(1)

    test_messages = [
        ("customer", "hi please I want 2 bags of rice and 1 garri, deliver to Bodija"),
        ("vendor", "restock garri, add 20"),
        ("vendor", "mark order 1 as delivered"),
        ("customer", "how much do you sell beans"),
        ("customer", "good morning"),
        ("vendor", "how much rice do we have left"),
    ]

    for role, msg in test_messages:
        print(f"\n> [{role}] {msg}")
        result = handle_message(role, msg)
        print(result["debug_log"])
        print(f"  [to customer] {result['customer_reply']}")