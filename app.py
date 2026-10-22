"""
app.py — the actual WhatsApp-facing server.

Meta sends every incoming message to POST /webhook. This pulls out the
sender + text (transcribing voice notes first if needed), decides if it's
the vendor or a customer, runs it through handle_message(), and sends the
result back over WhatsApp.

Duplicate protection: WhatsApp can redeliver the same webhook event (e.g.
if your server is slow to respond). Every message has a unique ID — we
track processed IDs in the DB and skip anything already handled, so a
redelivery never creates a second order.

Needs in .env (on top of GROQ_API_KEY):
    WHATSAPP_TOKEN=...
    WHATSAPP_PHONE_NUMBER_ID=...
    WHATSAPP_VERIFY_TOKEN=pick_any_string_yourself
    VENDOR_PHONE=2348012345678   # the vendor's own WhatsApp number, no '+'

Run locally with ngrok for testing:
    python3 app.py
    ngrok http 5000
Then put the ngrok https URL + /webhook as the Callback URL in Meta's
App Dashboard > WhatsApp > Configuration, with your VERIFY_TOKEN.
"""

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from agent_core_groq import handle_message
from whatsapp_client import send_whatsapp_text, download_whatsapp_media
from transcribe import transcribe_audio, is_trivial_transcription
from db import init_db, is_message_processed, mark_message_processed

load_dotenv()
init_db()

VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN")
VENDOR_PHONE = os.environ.get("VENDOR_PHONE")

app = Flask(__name__)


@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def receive_message():
    data = request.get_json(silent=True) or {}

    try:
        entry = data["entry"][0]
        change = entry["changes"][0]["value"]
        messages = change.get("messages")
        if not messages:
            return jsonify({"status": "ignored"}), 200

        message = messages[0]
        message_id = message.get("id")

        # Duplicate-delivery guard — do this before ANY processing
        if message_id and is_message_processed(message_id):
            print(f"[duplicate] skipping already-processed message {message_id}")
            return jsonify({"status": "duplicate_skipped"}), 200

        sender = message["from"]
        msg_type = message.get("type")
        is_voice = False

        if msg_type == "text":
            text = message["text"]["body"]
        elif msg_type == "audio":
            media_id = message["audio"]["id"]
            is_voice = True
            try:
                audio_bytes = download_whatsapp_media(media_id)
                text = transcribe_audio(audio_bytes)
                print(f"[voice note transcribed] {sender}: {text}")
                if is_trivial_transcription(text):
                    if message_id:
                        mark_message_processed(message_id)
                    send_whatsapp_text(sender, "Sorry, I couldn't make out that voice note — could you try again or type it instead?")
                    return jsonify({"status": "empty_transcription"}), 200
            except Exception as e:
                print(f"[voice note error] {e}")
                if message_id:
                    mark_message_processed(message_id)
                send_whatsapp_text(sender, "Sorry, I had trouble processing that voice note — could you type your message instead?")
                return jsonify({"status": "transcription_failed"}), 200
        else:
            if message_id:
                mark_message_processed(message_id)
            send_whatsapp_text(sender, "Sorry, I can only handle text or voice notes right now.")
            return jsonify({"status": "unsupported_message_type"}), 200

        # Mark processed BEFORE running the (slow) pipeline, so a redelivery
        # that arrives while we're still working doesn't slip through.
        if message_id:
            mark_message_processed(message_id)

        sender_role = "vendor" if sender == VENDOR_PHONE else "customer"
        result = handle_message(sender_role, text, sender_id=sender, is_voice=is_voice)
        print(f"[{sender_role}] {sender}: {text}\n  -> {result['debug_log']}")

        send_whatsapp_text(sender, result["customer_reply"])

    except Exception as e:
        print(f"[webhook error] {e}")

    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    if not VERIFY_TOKEN:
        print("Set WHATSAPP_VERIFY_TOKEN in .env before running.")
        raise SystemExit(1)
    app.run(port=5000, debug=True)