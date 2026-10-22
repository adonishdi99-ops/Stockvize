"""
transcribe.py — turns audio bytes (a WhatsApp voice note) into text
using Groq's free Whisper endpoint. Same GROQ_API_KEY as everything else.
"""

import io
from groq import Groq

client = Groq()

WHISPER_MODEL = "whisper-large-v3-turbo"  # fast + free tier, good enough for short voice notes


def transcribe_audio(audio_bytes: bytes, filename: str = "voice_note.ogg") -> str:
    """WhatsApp voice notes come as OGG/Opus — Groq's Whisper endpoint accepts that directly."""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename  # the SDK needs a filename to infer the format

    result = client.audio.transcriptions.create(
        file=audio_file,
        model=WHISPER_MODEL,
        response_format="text",
    )
    return str(result).strip()


def is_trivial_transcription(text: str) -> bool:
    """Catches near-silent audio that Whisper transcribes as just punctuation
    or a couple of stray characters instead of returning truly empty text."""
    stripped = text.strip(" .,!?-_\n\t")
    return len(stripped) < 2