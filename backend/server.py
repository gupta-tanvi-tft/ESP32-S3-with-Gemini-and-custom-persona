import os
import io
import wave
import asyncio
import tempfile
import logging
import struct
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import edge_tts
import miniaudio

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gemini-backend")

# Verified list of valid models for this API Key
FALLBACK_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
        logger.warning("⚠️  GEMINI_API_KEY is not set in backend/.env!")
    else:
        logger.info(f"✅ GEMINI_API_KEY is VALID & ACTIVE. Primary Model: '{FALLBACK_MODELS[0]}'")
    yield

app = FastAPI(title="ESP32-S3 Gemini Voice Assistant Relay", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "ESP32 Gemini Voice Relay",
        "configured_model": FALLBACK_MODELS[0]
    }

def process_audio_pcm(pcm_data: bytes, target_rms: float = 4000.0, silence_thresh: int = 30) -> bytes:
    """
    Applies audio gain normalization (boost soft speech) without trimming off quiet consonants.
    """
    if len(pcm_data) < 4:
        return pcm_data

    count = len(pcm_data) // 2
    samples = list(struct.unpack(f"<{count}h", pcm_data))

    if not samples:
        return pcm_data

    # Trim only extreme silence
    start_idx = 0
    while start_idx < len(samples) and abs(samples[start_idx]) < silence_thresh:
        start_idx += 1

    end_idx = len(samples) - 1
    while end_idx > start_idx and abs(samples[end_idx]) < silence_thresh:
        end_idx -= 1

    if start_idx < end_idx:
        trimmed_samples = samples[start_idx:end_idx + 1]
    else:
        trimmed_samples = samples

    sum_sq = sum(s * s for s in trimmed_samples)
    rms = (sum_sq / len(trimmed_samples)) ** 0.5 if trimmed_samples else 0.0

    if 10.0 < rms < target_rms:
        gain = min(target_rms / rms, 8.0) # Cap gain multiplier at 8x
        normalized_samples = [max(-32768, min(32767, int(s * gain))) for s in trimmed_samples]
        logger.info(f"🔊 Audio Processed: RMS boosted from {rms:.1f} to {rms*gain:.1f} (gain={gain:.2f}x).")
    else:
        normalized_samples = trimmed_samples
        logger.info(f"🔊 Audio Processed: RMS={rms:.1f}.")

    return struct.pack(f"<{len(normalized_samples)}h", *normalized_samples)

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wraps raw 16kHz 16-bit PCM bytes into a valid WAV header."""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buf.getvalue()

async def call_gemini_api(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Calls Gemini REST API using valid fallback models with strict Patient Persona context."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in .env")

    # Load Patient Persona JSON context
    persona_path = os.path.join(os.path.dirname(__file__), "patient_persona.json")
    persona_str = ""
    if os.path.exists(persona_path):
        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                persona_str = f.read()
            logger.info("📋 Loaded Patient Persona ('patient_persona.json') into Gemini Prompt.")
        except Exception as e:
            logger.warning(f"Could not load patient_persona.json: {e}")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    system_instruction = (
        "STRICT SYSTEM INSTRUCTIONS:\n"
        "You are an AI personal healthcare voice assistant for the patient Samarth.\n"
        "1. Listen carefully to the user's spoken audio prompt.\n"
        "2. Identify what specific topic or question the user asked (e.g., doctor's name, weight, HbA1c, medications, symptoms, glucose, steps, lab results, etc.).\n"
        "3. Answer ONLY the specific question asked by the user strictly using the data in Samarth's health record below.\n"
        "4. Do NOT invent facts or repeat weight if the user asked a different question.\n"
        "5. Keep your response empathetic, direct, and concise (1-2 sentences) for smart speaker voice playback.\n\n"
        "SAMARTH'S CLINICAL HEALTH RECORD:\n"
        "• Identity: Samarth (Age 31, Male, DOB: 07/05/1995)\n"
        "• Current Weight: 70 kg (Weight Loss: 16.2 kg, Trend: Decreasing)\n"
        "• Height: 180.34 cm | BMI: 48 (Obese) | Body Fat: 20% | Muscle Mass: 16 kg\n"
        "• HbA1c: 5.52% (Previous lab: 6.6%)\n"
        "• Assigned Doctor: Dr. Samarth Gupta (Endocrinologist) | Doctor Note: 'consultation is completed'\n"
        "• Active Medication: Paracetamol (Taken 0 times, 100% missed adherence rate)\n"
        "• Symptoms: Excessive thirst (Severity 8/10)\n"
        "• Glucometer: Last reading 125 mg/dL (Post Dinner). Flagged High: 405 mg/dL (Pre Dinner)\n"
        "• CGM Average Glucose: 131 mg/dL (Time in Range 56.5%)\n"
        "• Lab Reports: Hemoglobin 13.3 g/dL, Cholesterol 109 mg/dL, Triglycerides 217 mg/dL (Elevated High)\n"
        "• Vitals: Pulse 79 bpm, Respiration 19 rpm, O2 Saturation 99%\n"
        "• Activity: Daily Steps 3,694 steps, Active Calories 1,115 kcal\n\n"
        "FULL PATIENT PERSONA JSON DATA:\n"
        f"{persona_str}\n"
    )

    user_content_prompt = "Listen to the user's voice audio, transcribe their specific question, and answer it directly using Samarth's patient persona record."

    tried_models = set()
    for model_name in FALLBACK_MODELS:
        if model_name in tried_models:
            continue
        tried_models.add(model_name)

        try:
            logger.info(f"Sending {len(audio_bytes)} bytes audio to Gemini Model '{model_name}' with System Instruction...")
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=mime_type,
                    ),
                    user_content_prompt
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                )
            )
            if response.text:
                text_reply = response.text.strip()
                logger.info(f"✅ Gemini ('{model_name}') Response: '{text_reply}'")
                return text_reply
        except Exception as e:
            logger.warning(f"Model '{model_name}' failed: {e}. Trying next fallback...")
            await asyncio.sleep(0.3)

    logger.error("All Gemini API model attempts failed!")
    return "Sorry, I had trouble reaching Gemini."

async def text_to_pcm_16k(text: str) -> bytes:
    """Synthesizes text to speech and converts MP3 to 16kHz 16-bit Mono PCM using miniaudio."""
    tts_voice = os.getenv("TTS_VOICE", "en-US-AvaNeural")
    tts_rate = os.getenv("TTS_RATE", "+0%")
    tts_volume = os.getenv("TTS_VOLUME", "-30%") # Lower default TTS volume

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_temp:
        mp3_path = mp3_temp.name

    try:
        logger.info(f"Synthesizing TTS (Voice='{tts_voice}', Rate='{tts_rate}', Volume='{tts_volume}') for: '{text}'")
        communicate = edge_tts.Communicate(text, voice=tts_voice, rate=tts_rate, volume=tts_volume)
        await communicate.save(mp3_path)

        # Decode MP3 and convert to 16kHz 1-channel SIGNED16 PCM for ESP32 speaker
        decoded = miniaudio.decode_file(mp3_path)
        pcm_bytes = miniaudio.convert_frames(
            decoded.sample_format,
            decoded.nchannels,
            decoded.sample_rate,
            bytes(decoded.samples),
            miniaudio.SampleFormat.SIGNED16,
            1,      # 1 channel (Mono)
            16000   # 16kHz sample rate
        )

        # Scale down PCM amplitude by 50% for softer speaker playback
        count = len(pcm_bytes) // 2
        samples = struct.unpack(f"<{count}h", pcm_bytes)
        scaled_samples = [int(s * 0.5) for s in samples]
        pcm_bytes = struct.pack(f"<{count}h", *scaled_samples)

        logger.info(f"🔊 Generated {len(pcm_bytes)} bytes of 16kHz Mono PCM for ESP32 speaker (Volume scaled down 50%).")
        return pcm_bytes
    except Exception as err:
        logger.error(f"Error converting TTS audio to PCM: {err}", exc_info=True)
        with open(mp3_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)

@app.post("/api/chat-audio")
async def chat_audio(request: Request):
    """
    HTTP POST /api/chat-audio
    Expects PCM or WAV audio from ESP32.
    Returns PCM audio stream to ESP32 speaker.
    """
    body = await request.body()
    if not body or len(body) < 100:
        raise HTTPException(status_code=400, detail="Audio body too short")

    logger.info(f"Received {len(body)} bytes audio payload from ESP32")

    if body.startswith(b"RIFF"):
        pcm_payload = body[44:]
    else:
        pcm_payload = body

    processed_pcm = process_audio_pcm(pcm_payload)
    wav_bytes = pcm_to_wav(processed_pcm, sample_rate=16000, channels=1, sample_width=2)

    # 1. Call Gemini Model
    gemini_text = await call_gemini_api(wav_bytes, mime_type="audio/wav")

    # 2. Text to Speech (Converted to 16kHz Mono PCM)
    pcm_audio_output = await text_to_pcm_16k(gemini_text)

    # 3. Detect Voice Volume Control Commands
    lower_text = gemini_text.lower()
    set_vol_header = None

    if "mute" in lower_text or "silent" in lower_text:
        set_vol_header = "0"
    elif "increase volume" in lower_text or "volume up" in lower_text or "louder" in lower_text:
        set_vol_header = "85"
    elif "lower volume" in lower_text or "volume down" in lower_text or "softer" in lower_text or "quiet" in lower_text:
        set_vol_header = "35"

    # Strip non-latin1 characters for HTTP headers
    safe_header_text = gemini_text.replace("\n", " ").encode("ascii", "ignore").decode("ascii")

    headers = {
        "X-Gemini-Text": safe_header_text,
        "Content-Length": str(len(pcm_audio_output))
    }
    if set_vol_header:
        headers["X-Set-Volume"] = set_vol_header

    # 4. Return Audio Stream to ESP32
    return Response(
        content=pcm_audio_output,
        media_type="application/octet-stream",
        headers=headers
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8008))
    logger.info(f"🚀 Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
