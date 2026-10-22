"""
whatsapp_client.py — thin wrapper around Meta's WhatsApp Cloud API
for sending text messages and downloading media (voice notes).

Needs in .env:
    WHATSAPP_TOKEN=your_access_token
    WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")

GRAPH_API_VERSION = "v21.0"
BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"


def download_whatsapp_media(media_id: str) -> bytes:
    """WhatsApp gives you a media ID, not a direct URL. Two steps: look up the
    real (temporary) URL, then download the actual bytes from it."""
    if not WHATSAPP_TOKEN:
        raise RuntimeError("WHATSAPP_TOKEN missing from .env")

    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    lookup_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{media_id}"
    resp = requests.get(lookup_url, headers=headers, timeout=15)
    resp.raise_for_status()
    media_url = resp.json()["url"]

    media_resp = requests.get(media_url, headers=headers, timeout=30)
    media_resp.raise_for_status()
    return media_resp.content


def send_whatsapp_text(to: str, body: str) -> dict:
    """Send a plain text message to a WhatsApp number (E.164 format, no '+')."""
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        raise RuntimeError("WHATSAPP_TOKEN or WHATSAPP_PHONE_NUMBER_ID missing from .env")

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=15)
    if resp.status_code >= 400:
        print(f"[whatsapp_client] send failed ({resp.status_code}): {resp.text}")
    resp.raise_for_status()
    return resp.json()