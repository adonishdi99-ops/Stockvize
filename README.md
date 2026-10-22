# Stockvize

An AI agent that lets a small shop run its orders and inventory over **WhatsApp**. Customers (and the vendor) message the shop in plain, messy language, including Nigerian Pidgin, typos, and voice notes. Stockvize turns each message into a real action: placing an order, updating stock, or answering a question.

## Features

- **WhatsApp-native**: receives messages through Meta's WhatsApp Cloud API webhook
- **Voice note support**: transcribes voice notes with Whisper (via Groq) before processing
- **LLM tool-calling brain**: the model picks a structured tool instead of free-typing replies
- **Customer vs vendor roles**: the vendor's number can restock, correct stock, and update order status; everyone else is treated as a customer
- **Messy-language handling**: normalizes typos and slang (e.g. "rise" to "rice") and asks for clarification on garbled voice-note locations
- **Conversation memory**: recent turns per sender are kept, so follow-ups like "I said Ikeja" fix the existing order instead of creating a new one
- **Duplicate protection**: WhatsApp message IDs are tracked, so a redelivered webhook never creates a second order
- **SQLite storage**: inventory, orders, processed messages, and conversation history

## How It Works

```
WhatsApp message (text or voice note)
        |
        v
  Flask webhook  (app.py)
        |-- voice note? --> download media --> Whisper transcription
        v
  Agent brain  (agent_core_groq.py)
   LLM picks one or more tools:
     create_order / update_inventory / get_stock_level
     mark_order_status / update_order_location
     get_last_order / reply_customer
        |
        v
  SQLite database  (db.py)
        |
        v
  Reply sent back over WhatsApp  (whatsapp_client.py)
```

## Tech Stack

- Python 3
- Flask
- Groq API (tool-calling LLM and Whisper transcription)
- SQLite
- Meta WhatsApp Cloud API

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/stockvize.git
cd stockvize

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # macOS / Linux
```

Then fill in `.env`:

| Variable | Description |
| --- | --- |
| `GROQ_API_KEY` | Free key from [console.groq.com](https://console.groq.com) |
| `WHATSAPP_TOKEN` | Access token from Meta's App Dashboard |
| `WHATSAPP_PHONE_NUMBER_ID` | Phone number ID from WhatsApp > API Setup |
| `WHATSAPP_VERIFY_TOKEN` | Any string you choose; Meta sends it back to verify your webhook |
| `VENDOR_PHONE` | The vendor's own WhatsApp number, digits only, no `+` |

### 3. Create the database

```bash
python db.py
```

This creates `vendor_agent.db` and seeds starter stock (rice, garri, beans).

## Running

Start the server:

```bash
python app.py
```

Expose it to the internet for testing (for example with [ngrok](https://ngrok.com)):

```bash
ngrok http 5000
```

In Meta's App Dashboard, go to **WhatsApp > Configuration** and set:

- **Callback URL:** `https://<your-ngrok-url>/webhook`
- **Verify token:** the same value as `WHATSAPP_VERIFY_TOKEN`

## Testing Without WhatsApp

You can test the agent's brain directly from the terminal:

```bash
python agent_core_groq.py    # basic scenarios
python stress_test.py        # messier, pidgin-style messages and edge cases
```

`test_payload.example.json` shows the shape of a WhatsApp send-message payload. Copy it and put in your own test number.

## Project Structure

```
stockvize/
├── app.py                  # Flask webhook: receives and routes WhatsApp messages
├── agent_core_groq.py      # LLM brain: tool schema, system prompt, message pipeline
├── db.py                   # SQLite storage and helper functions
├── whatsapp_client.py      # Send messages / download media via WhatsApp Cloud API
├── transcribe.py           # Voice note transcription (Groq Whisper)
├── stress_test.py          # Edge-case test messages for the agent
├── test_payload.example.json
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Notes and Limitations

- The shop's items are currently hard-coded in `KNOWN_ITEMS` (`agent_core_groq.py`). Replace them with the real shop's stock list.
- `app.py` runs Flask with `debug=True`, which is fine for local testing. Turn it off and use a production server (e.g. gunicorn) before deploying.
- The agent has no price list yet, so it won't quote prices.
- Never commit `.env` or `vendor_agent.db`. The database holds real customer phone numbers and conversations.

## Roadmap

- [ ] Price list and order totals
- [ ] Vendor daily summary of orders and low stock
- [ ] Multi-vendor support
- [ ] Production deployment setup

## License

Released under the [MIT License](LICENSE).

## Author

**Adonis**, Software Engineering student at FUNAAB, Nigeria.