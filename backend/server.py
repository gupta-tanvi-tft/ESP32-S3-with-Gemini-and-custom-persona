import os
import io
import wave
import math
import asyncio
import tempfile
import logging
import struct
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gemini-backend")

PRIMARY_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
        logger.warning("⚠️  GEMINI_API_KEY is not set in backend/.env!")
    else:
        logger.info(f"✅ GEMINI_API_KEY is VALID & ACTIVE. Live Model: '{PRIMARY_LIVE_MODEL}'")
    yield

app = FastAPI(title="ESP32-S3 Gemini Voice Assistant", lifespan=lifespan)

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
        "configured_model": PRIMARY_LIVE_MODEL
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

async def call_gemini_api(audio_bytes: bytes, mime_type: str = "audio/wav") -> dict:
    """
    Analyzes user spoken audio with Gemini 3.1:
    1. Transcribes spoken question
    2. Performs emotion & intent detection from audio tone
    3. Derives human-like clinical response strictly from patient record
    4. Determines target voice emotion (celebratory, calm_reassuring, empathetic_gentle, warm_clinical)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in .env")

    # Load Patient Persona JSON context
    persona_path = os.path.join(os.path.dirname(__file__), "patient_persona.json")
    persona_str = ""
    patient_name = "the patient"
    
    if os.path.exists(persona_path):
        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                persona_str = f.read()
            import json
            p_data = json.loads(persona_str).get("data", {})
            identity = p_data.get("identity", {})
            patient_name = identity.get("first_name", "the patient")
        except Exception as e:
            logger.warning(f"Could not load patient_persona.json: {e}")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    system_instruction = (
        "STRICT HUMAN VOICE INTELLIGENCE & CLINICAL ASSISTANT INSTRUCTIONS:\n"
        f"You are a human-like, highly empathetic clinical voice companion speaking directly to {patient_name}.\n"
        "Analyze both the spoken content AND the tone of the user's voice.\n"
        "1. IF THE USER SAYS A GREETING OR CASUAL CHITCHAT (e.g., 'Hello', 'Hi', 'Hey', 'Good morning', 'How are you'):\n"
        "   - Greet them back warmly and naturally by name! Ask how they are feeling today.\n"
        "   - DO NOT blurt out clinical data or HbA1c/medical metrics unless specifically asked!\n"
        "2. IF THE USER ASKS A SPECIFIC HEALTH QUESTION:\n"
        "   - Search the PATIENT PERSONA RECORD below and extract exact values (doctor name/ID, HbA1c, glucose, vitals, step count, medications, lab reports, doctor notes, etc.).\n"
        "3. Determine the user's emotion/tone (`happy`, `anxious`, `concerned`, `pain`, `neutral`, `curious`).\n"
        "4. Determine the best voice response mood for speech synthesis (`celebratory`, `calm_reassuring`, `empathetic_gentle`, `warm_clinical`).\n"
        "   - Use `celebratory` for positive achievements or friendly greetings.\n"
        "   - Use `calm_reassuring` for user anxiety, elevated glucose spikes, or high blood pressure.\n"
        "   - Use `empathetic_gentle` for pain, discomfort, or missed medication notes.\n"
        "   - Use `warm_clinical` for general informative questions.\n"
        "5. Respond STRICTLY in valid JSON format:\n"
        "{\n"
        '  "transcription": "<exact transcribed user question>",\n'
        '  "user_emotion": "<detected emotion>",\n'
        '  "response_mood": "<celebratory | calm_reassuring | empathetic_gentle | warm_clinical>",\n'
        '  "answer": "<warm, natural, 1-2 sentence conversational response>"\n'
        "}\n\n"
        "PATIENT PERSONA RECORD:\n"
        f"{persona_str}\n"
    )

    user_content_prompt = f"Listen to the spoken audio, perceive user tone, transcribe as transcription, determine response_mood, and answer as answer in JSON format."

    try:
        logger.info(f"Sending {len(audio_bytes)} bytes audio to Gemini Live Model '{PRIMARY_LIVE_MODEL}' for Human Voice & Persona Reasoning...")
        response = client.models.generate_content(
            model=PRIMARY_LIVE_MODEL,
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
            import json
            raw_text = response.text.strip()
            # Clean markdown json fences if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            try:
                res_json = json.loads(raw_text)
                transcription = res_json.get("transcription", "")
                
                # Detect garbled/unclear speech or empty transcription
                if not transcription or transcription.strip().lower() in ["", "[unclear]", "[inaudible]", "noise", "thank you", "sound", "voice query"]:
                    res_json["answer"] = "I didn't quite catch that. Could you please speak clearly?"
                    res_json["response_mood"] = "empathetic_gentle"

                logger.info(f" 🗣️ User Spoke: '{res_json.get('transcription')}' [User Emotion: {res_json.get('user_emotion')}]")
                logger.info(f" 🧠 Gemini Response Mood: '{res_json.get('response_mood')}'")
                logger.info(f" 💬 Gemini Answer: '{res_json.get('answer')}'")
                return res_json
            except Exception:
                logger.info(f" Gemini Text (Non-JSON fallback): '{raw_text}'")
                return {
                    "transcription": "Voice Query",
                    "user_emotion": "neutral",
                    "response_mood": "warm_clinical",
                    "answer": raw_text
                }
    except Exception as e:
        logger.error(f"Gemini API model '{PRIMARY_MODEL}' error: {e}")

    return {
        "transcription": "Error",
        "user_emotion": "concerned",
        "response_mood": "empathetic_gentle",
        "answer": "I didn't catch that. Could you please repeat and speak clearly?"
    }

async def text_to_pcm_16k(text: str, mood: str = "warm_clinical") -> bytes:
    """
    Synthesizes speech using mood-adaptive SSML voice parameters (pitch, rate, volume)
    and converts MP3 to 16kHz 16-bit Mono PCM for ESP32 playback.
    """
    base_voice = os.getenv("TTS_VOICE", "en-US-AvaNeural")
    
    # Configure dynamic SSML voice parameters based on emotional mood
    if mood == "celebratory":
        rate = "+25%"
        pitch = "+4Hz"
        volume = "+5%"
        style = "cheerful"
    elif mood == "calm_reassuring":
        rate = "+12%"
        pitch = "+1Hz"
        volume = "+0%"
        style = "friendly"
    elif mood == "empathetic_gentle":
        rate = "+15%"
        pitch = "+2Hz"
        volume = "-2%"
        style = "empathetic"
    else: # warm_clinical / conversational default
        rate = "+20%"
        pitch = "+3Hz"
        volume = "+0%"
        style = "cheerful"

    # Build SSML string for human expressive vocal contour
    ssml_text = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
    <voice name='{base_voice}'>
        <prosody rate='{rate}' pitch='{pitch}' volume='{volume}'>
            {text}
        </prosody>
    </voice>
</speak>"""

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_temp:
        mp3_path = mp3_temp.name

    try:
        logger.info(f"Synthesizing Dynamic SSML Voice [Mood='{mood}', Voice='{base_voice}', Rate='{rate}', Pitch='{pitch}']")
        communicate = edge_tts.Communicate(ssml_text, voice=base_voice)
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

        # Apply energetic amplitude scaling for crisp playback
        count = len(pcm_bytes) // 2
        samples = struct.unpack(f"<{count}h", pcm_bytes)
        scaled_samples = [max(-32768, min(32767, int(s * 0.85))) for s in samples]
        pcm_bytes = struct.pack(f"<{count}h", *scaled_samples)

        logger.info(f" Generated {len(pcm_bytes)} bytes of mood-modulated 16kHz PCM audio.")
        return pcm_bytes
    except Exception as err:
        logger.error(f"Error converting TTS audio to PCM: {err}", exc_info=True)
        # Fallback to plain text TTS if SSML fails
        try:
            communicate = edge_tts.Communicate(text, voice=base_voice)
            await communicate.save(mp3_path)
            decoded = miniaudio.decode_file(mp3_path)
            return miniaudio.convert_frames(
                decoded.sample_format, decoded.nchannels, decoded.sample_rate,
                bytes(decoded.samples), miniaudio.SampleFormat.SIGNED16, 1, 16000
            )
        except Exception:
            return b""
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

    # 1. Call Gemini Model API (Perceives user emotion & derives persona answer)
    gemini_res = await call_gemini_api(wav_bytes, mime_type="audio/wav")
    answer_text = gemini_res.get("answer", "I'm here to help you.")
    response_mood = gemini_res.get("response_mood", "warm_clinical")

    # 2. Text to Speech (Converted to 16kHz Mono PCM with SSML voice mood)
    pcm_audio_output = await text_to_pcm_16k(answer_text, mood=response_mood)

    # 3. Detect Voice Volume Control Commands
    lower_text = answer_text.lower()
    set_vol_header = None

    if "mute" in lower_text or "silent" in lower_text:
        set_vol_header = "0"
    elif "increase volume" in lower_text or "volume up" in lower_text or "louder" in lower_text:
        set_vol_header = "85"
    elif "lower volume" in lower_text or "volume down" in lower_text or "softer" in lower_text or "quiet" in lower_text:
        set_vol_header = "35"

    # Strip non-latin1 characters for HTTP headers
    safe_header_text = answer_text.replace("\n", " ").encode("ascii", "ignore").decode("ascii")

    headers = {
        "X-Gemini-Text": safe_header_text,
        "X-Response-Mood": response_mood,
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

class SmoothResampler24kTo16k:
    """
    Studio-Grade 24kHz -> 16kHz Streaming Polyphase Resampler.
    Uses continuous 4-point symmetric cubic interpolation with matched zero-phase filtering
    and boundary sample history to eliminate 8kHz Nyquist modulation flutter, robotic buzz, and aliasing distortion.
    """
    def __init__(self, volume_scale: float = 0.44):
        self.raw_bytes = bytearray()
        self.history = [0, 0, 0] # Holds last 3 input samples for seamless inter-chunk continuity
        self.remainder_samples = []
        self.last_out_sample = 0
        self.volume_scale = volume_scale

    def process(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""
        self.raw_bytes.extend(chunk)
        
        # Ensure we only process whole 16-bit samples (even number of bytes)
        num_samples = len(self.raw_bytes) // 2
        if num_samples == 0:
            return b""
            
        usable_bytes = num_samples * 2
        chunk_to_unpack = bytes(self.raw_bytes[:usable_bytes])
        self.raw_bytes = self.raw_bytes[usable_bytes:]
        
        new_samples = list(struct.unpack(f"<{num_samples}h", chunk_to_unpack))
        all_samples = self.remainder_samples + new_samples
        
        num_triplets = len(all_samples) // 3
        if num_triplets == 0:
            self.remainder_samples = all_samples
            return b""
            
        used_len = num_triplets * 3
        to_process = all_samples[:used_len]
        self.remainder_samples = all_samples[used_len:]
        
        # Build contiguous sequence with 3-sample history prefix
        seq = self.history + to_process
        self.history = to_process[-3:]
        
        out_samples = []
        scale = self.volume_scale
        for i in range(num_triplets):
            idx = 3 * i + 3
            sm1 = seq[idx - 1]
            s0  = seq[idx]
            s1  = seq[idx + 1]
            s2  = seq[idx + 2]
            s3  = seq[idx + 3] if (idx + 3) < len(seq) else s2
            
            # Symmetrically matched 4-point cubic Hermite interpolation:
            # y0 (on-grid sample): matched symmetric smoothing (sm1 + 14*s0 + s1) / 16
            # y1 (midpoint sample): cubic interpolated (-s0 + 9*s1 + 9*s2 - s3) / 16
            y0_raw = (sm1 + 14 * s0 + s1 + 8) >> 4
            y1_raw = (-s0 + 9 * s1 + 9 * s2 - s3 + 8) >> 4
            
            y0 = max(-32768, min(32767, int(y0_raw * scale)))
            y1 = max(-32768, min(32767, int(y1_raw * scale)))
            
            out_samples.append(y0)
            out_samples.append(y1)
            
        if out_samples:
            self.last_out_sample = out_samples[-1]
            
        return struct.pack(f"<{len(out_samples)}h", *out_samples)

    def flush(self) -> bytes:
        out_samples = []
        scale = self.volume_scale
        if len(self.remainder_samples) == 1:
            out_samples.append(int(self.remainder_samples[0] * scale))
        elif len(self.remainder_samples) == 2:
            out_samples.append(int(self.remainder_samples[0] * scale))
            out_samples.append(int(self.remainder_samples[1] * scale))
            
        # Smooth 32-sample linear fade-out to zero (anti-pop window)
        start_val = out_samples[-1] if out_samples else self.last_out_sample
        if abs(start_val) > 10:
            for k in range(1, 33):
                factor = (32 - k) / 32.0
                out_samples.append(int(start_val * factor))
                
        # 128 samples (~8ms) of clean zero silence to let DAC DMA drain smoothly
        out_samples.extend([0] * 128)
        
        self.raw_bytes.clear()
        self.history = [0, 0, 0]
        self.remainder_samples = []
        self.last_out_sample = 0
        return struct.pack(f"<{len(out_samples)}h", *out_samples) if out_samples else b""


@app.websocket("/ws/live/{session_id}")
@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket, session_id: str = "default"):
    """
    Real-Time Gemini Live API Bi-Directional WebSocket Endpoint.
    Streams continuous 16kHz PCM audio from ESP32 -> Gemini Live -> ESP32 speaker.
    Supports real-time barge-in interruption and persistent session state.
    """
    await websocket.accept()
    logger.info(f"🟢 Client connected to Gemini Live Stream endpoint (session_id: '{session_id}').")

    api_key = os.getenv("GEMINI_API_KEY")
    api_base_url = os.getenv("PATIENT_API_BASE_URL", "http://72.61.241.48:8000")
    bearer_token = os.getenv("PERSONA_API_TOKEN", "")

    persona_str = ""
    patient_name = "Patient"

    import json
    import urllib.request

    # Helper to resolve patient persona dynamically for tool calls
    async def load_persona_data_for_name(name_query: str):
        target_id = None
        matched_name = name_query
        try:
            doc_url = f"{api_base_url}/agent/patients/{session_id}"
            req = urllib.request.Request(doc_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                doc_data = json.loads(resp.read().decode("utf-8"))
                patients_list = doc_data.get("patients", [])
                query_clean = name_query.strip().lower()
                for p in patients_list:
                    fn = p.get("first_name", "").strip().lower()
                    ln = p.get("last_name", "").strip().lower()
                    full_n = f"{fn} {ln}".strip()
                    if query_clean in full_n or query_clean in fn or (fn and fn in query_clean):
                        target_id = p.get("id") or p.get("_id")
                        matched_name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                        logger.info(f"⚡ Patient Match: '{name_query}' -> ID: {target_id} ({matched_name})")
                        break
        except Exception as err:
            logger.warning(f"Error searching patient roster: {err}")

        p_info = {}
        if target_id:
            try:
                persona_url = f"{api_base_url}/persona/{target_id}"
                req = urllib.request.Request(persona_url, headers={
                    "Authorization": f"Bearer {bearer_token}",
                    "Accept": "application/json"
                })
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    p_info = data.get("persona", {})
            except Exception as err:
                logger.warning(f"Error fetching persona for ID '{target_id}': {err}")
        return matched_name, p_info

    # 1. Fetch Doctor's Patient List & Dynamic Patient Persona via APIs
    if session_id and session_id != "default":
        # Check if session_id is a Doctor ID (starts with 6788 or matches doctor ID length)
        try:
            doc_url = f"{api_base_url}/agent/patients/{session_id}"
            req = urllib.request.Request(doc_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                doc_data = json.loads(resp.read().decode("utf-8"))
                total_patients = doc_data.get("total", 0)
                patients_list = doc_data.get("patients", [])
                if total_patients > 0:
                    persona_str += f"\nDOCTOR PATIENTS DIRECTORY (Total: {total_patients} registered patients for Doctor ID '{session_id}'):\n" + json.dumps(patients_list, indent=2)
                    patient_name = "Doctor"
                    logger.info(f"✅ Loaded Doctor patient directory from API for Doctor ID '{session_id}' ({total_patients} patients)")
        except Exception as doc_err:
            logger.debug(f"Session ID '{session_id}' not a doctor ID or Doctor API error: {doc_err}")

        # Fetch specific Patient Persona if session_id is a Patient ID
        try:
            persona_url = f"{api_base_url}/persona/{session_id}"
            req = urllib.request.Request(persona_url, headers={
                "Authorization": f"Bearer {bearer_token}",
                "Accept": "application/json"
            })
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                p_data = data.get("persona", {})
                if p_data:
                    identity = p_data.get("identity", {})
                    patient_name = identity.get("first_name", patient_name)
                    persona_str += f"\nACTIVE PATIENT CLINICAL PERSONA:\n" + json.dumps(p_data, indent=2)
                    logger.info(f"✅ Loaded active patient persona from API for Patient ID '{session_id}' (Name: '{patient_name}')")
        except Exception as api_err:
            logger.debug(f"Session ID '{session_id}' persona API fetch info: {api_err}")

    # Fallback to local patient_persona.json if API was not loaded
    if not persona_str:
        persona_path = os.path.join(os.path.dirname(__file__), "patient_persona.json")
        if os.path.exists(persona_path):
            try:
                with open(persona_path, "r", encoding="utf-8") as f:
                    raw_json = f.read()
                full_dict = json.loads(raw_json)
                p_data = full_dict.get("data", {})
                identity = p_data.get("identity", {})
                patient_name = identity.get("first_name", "Samarth")
                copilot_ctx = p_data.get("ai_copilot_context", {})
                exec_summary = copilot_ctx.get("executive_summary_for_llm", "")
                persona_str = f"EXECUTIVE SUMMARY: {exec_summary}\nFULL DATA RECORD:\n" + json.dumps(p_data, indent=2)
                logger.info(f"✅ Loaded local patient persona from patient_persona.json (Patient Name: '{patient_name}')")
            except Exception as e:
                logger.warning(f"Could not load patient_persona.json: {e}")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1alpha"))

    doctor_name = "Samarth"
    specialization = "Endocrinologist"

    active_patient_name = patient_name if (patient_name and patient_name != "Patient" and patient_name != "Doctor") else None

    if active_patient_name and persona_str:
        patient_block = (
            f"ACTIVE PATIENT: {active_patient_name}\n"
            "A patient record is currently loaded. Answer clinical questions using ONLY the data below. Do not invent, estimate, or assume any value not present in this record.\n\n"
            "PATIENT PERSONA RECORD:\n"
            f"{persona_str}\n"
        )
        patient_state_rule = (
            "3. THE DOCTOR ASKS ABOUT THE ACTIVE PATIENT (vitals, glucose, HbA1c, blood pressure, weight, medications, adherence, lab results, diet or lifestyle recommendations, doctor's notes, appointments, or any other clinical detail):\n"
            "   - Answer directly from the PATIENT PERSONA RECORD above. Cite the actual number, date, or name — never a rounded or invented approximation.\n"
            "   - Keep it concise: 1-3 clear sentences, unless the doctor asks for a fuller summary.\n"
            f"   - If the record does not contain the answer, say so plainly: 'That is not in the current record for {active_patient_name}.' Never fill the gap with a guess.\n"
            "   - If a value in the record falls in an emergency range (glucose below 54 or above 350 mg/dL, systolic BP above 180, diastolic above 120), lead with that flag before anything else, in a clear and direct — not alarmed — tone.\n"
            "   - Offer interpretation as data-grounded observation, not as a diagnosis: describe what the readings show and how they compare to target, and let the doctor draw the clinical conclusion. You are surfacing information for a licensed clinician, not replacing their judgment.\n"
        )
    else:
        patient_block = (
            "ACTIVE PATIENT: none loaded yet.\n"
            "No patient record is currently loaded. Do not answer any clinical question until a patient has been loaded via the load_patient_record tool.\n"
        )
        patient_state_rule = (
            "3. THE DOCTOR HAS NOT YET NAMED A PATIENT:\n"
            "   - If the doctor asks a clinical question before naming a patient, ask once, briefly: 'Which patient would you like to discuss?' Do not guess who they mean.\n"
        )

    system_instruction = (
        "STRICT HUMAN VOICE INTELLIGENCE & CLINICAL ASSISTANT INSTRUCTIONS:\n"
        f"You are YHealth Assist, a clinical AI coordinator speaking directly with Dr. {doctor_name}, {specialization}. You support the doctor's clinical workflow by retrieving and discussing patient records, hands-free, during their consultations.\n"
        "Maintain a clear, confident, efficient conversational pace — this is a working clinical tool for a doctor moving quickly between patients, not a leisurely chat.\n"
        "Speak naturally and professionally, without rushing or sounding clipped, and without shouting or abrupt tone changes.\n"
        "CRITICAL: Never append repetitive boilerplate or disclaimers (e.g. 'please verify independently', 'this is not medical advice') to routine answers. The doctor is the licensed clinician here — give direct, grounded answers only.\n\n"
        "CONVERSATION RULES:\n"
        f"1. WHEN THE DOCTOR CALLS YOUR WAKE WORD ('Hey YHealth', 'Hello YHealth Assist', 'Hi Assistant') OR GREETS YOU:\n"
        f"   - Greet Dr. {doctor_name} warmly but briefly by name and title (e.g. 'Good morning, Dr. {doctor_name}. Which patient would you like to start with?').\n"
        "   - Keep it to one sentence. Do not list capabilities unless asked.\n"
        "2. WHEN THE DOCTOR STATES OR SWITCHES A PATIENT NAME (at session start, or at any point mid-conversation, even while another patient's record is already loaded):\n"
        "   - Call load_patient_record with the spoken name immediately. Do not answer clinical questions from memory of a prior patient once a new name has been stated.\n"
        "   - While waiting on the tool response, give a brief holding line: 'One moment, pulling up that record.'\n"
        "   - If the backend cannot find a matching patient, say so plainly and ask the doctor to repeat or confirm the name — never fabricate a record.\n"
        f"{patient_state_rule}"
        "4. WHEN THE DOCTOR DICTATES A CLINICAL NOTE (e.g. 'note for [patient]: increase walking to 45 minutes daily'):\n"
        "   - Repeat the note content back once for confirmation before calling add_clinical_note.\n"
        "   - After the tool confirms, say: 'Note saved.' Nothing more.\n"
        "5. WHEN THE DOCTOR ASKS TO SEND AN ALERT OR NOTIFY A PATIENT:\n"
        "   - Confirm the patient and alert type, then call escalate_alert.\n"
        "   - After the tool confirms, say: 'Alert sent.' Nothing more.\n"
        "6. WHEN THE DOCTOR SAYS 'STOP', 'GOODBYE', 'BYE', 'THAT IS ALL', OR 'GO TO SLEEP':\n"
        f"   - Say a brief, professional sign-off (e.g. 'Goodbye, Dr. {doctor_name}. Have a good day.') and conclude.\n\n"
        "CLINICAL SAFETY RULES — ABSOLUTE:\n"
        "- Every number, date, or name you speak must come from the PATIENT PERSONA RECORD. Never estimate, round beyond what is given, or infer a value that is not present.\n"
        "- Never suggest a specific medication dosage change, substitution, or discontinuation — that decision belongs to the doctor. You may state what the record shows (e.g. adherence, current prescription) but not what the doctor should prescribe.\n"
        "- Never state a diagnosis as settled fact. Describe what the data shows and how it compares to target ranges; let the doctor conclude.\n"
        "- If asked something outside the patient's record or outside your role, say plainly that it's outside what you can confirm from this record.\n\n"
        f"DOCTOR CONTEXT:\nName: Dr. {doctor_name}\nSpecialization: {specialization}\n\n"
        f"{patient_block}"
    )

    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="load_patient_record",
                    description="Call this the moment the doctor states, confirms, or switches to a patient's name — whether at the start of the session or mid-conversation. Do not attempt to answer clinical questions before calling this and receiving the patient record back.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "patient_name": types.Schema(
                                type=types.Type.STRING,
                                description="The patient name as spoken by the doctor, verbatim."
                            )
                        },
                        required=["patient_name"]
                    )
                ),
                types.FunctionDeclaration(
                    name="add_clinical_note",
                    description="Call this when the doctor explicitly dictates a clinical note to save against the currently loaded patient's record. Only call after the doctor confirms the note content is correct.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "patient_name": types.Schema(type=types.Type.STRING),
                            "note_content": types.Schema(type=types.Type.STRING)
                        },
                        required=["patient_name", "note_content"]
                    )
                ),
                types.FunctionDeclaration(
                    name="escalate_alert",
                    description="Call this when the doctor explicitly asks to send an alert or notification to the currently loaded patient.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "patient_name": types.Schema(type=types.Type.STRING),
                            "alert_type": types.Schema(
                                type=types.Type.STRING,
                                description="glucose_high, glucose_low, bp_high, medication_missed, general"
                            )
                        },
                        required=["patient_name", "alert_type"]
                    )
                )
            ]
        )
    ]

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=system_instruction)]
        ),
        tools=tools
    )

    try:
        logger.info(f"⚡ Establishing Gemini Live API Session [{PRIMARY_LIVE_MODEL}] for session_id='{session_id}'...")
        async with client.aio.live.connect(model=PRIMARY_LIVE_MODEL, config=config) as session:
            logger.info(f"✅ Gemini Live API Session ACTIVE [{PRIMARY_LIVE_MODEL}] for session_id='{session_id}'")

            loop = asyncio.get_event_loop()
            last_activity_time = loop.time()
            last_audio_in_time = 0.0
            streaming_speech = False
            is_model_speaking = False
            resampler = SmoothResampler24kTo16k(volume_scale=0.46)

            def boost_chunk_gain(raw_bytes: bytes, gain: float = 1.4) -> bytes:
                if len(raw_bytes) < 4:
                    return raw_bytes
                count = len(raw_bytes) // 2
                samples = struct.unpack(f"<{count}h", raw_bytes[:count*2])
                boosted = [max(-32768, min(32767, int(s * gain))) for s in samples]
                return struct.pack(f"<{count}h", *boosted)

            async def gemini_rx_loop():
                nonlocal last_activity_time, is_model_speaking
                try:
                    sent_bytes_in_turn = 0
                    turn_start_time = 0.0

                    async for response in session.receive():
                        last_activity_time = loop.time()

                        # Handle Tool Call from Gemini Live for API 2 Persona Lookup / Notes / Alerts
                        if response.tool_call is not None:
                            for call in response.tool_call.function_calls:
                                logger.info(f"🔍 Executing Tool Call: '{call.name}' args={call.args}")
                                matched_n = call.args.get("patient_name", "Patient")
                                p_info = {}
                                if call.name == "load_patient_record":
                                    matched_n, p_info = await load_persona_data_for_name(matched_n)
                                tool_response = types.LiveClientToolResponse(
                                    function_responses=[
                                        types.FunctionResponse(
                                            name=call.name,
                                            id=call.id,
                                            response={"status": "success", "matched_patient": matched_n, "patient_data": p_info}
                                        )
                                    ]
                                )
                                try:
                                    await session.send_tool_response(function_responses=tool_response.function_responses)
                                except Exception:
                                    await session.send(input=tool_response)
                        server_content = response.server_content
                        if server_content is not None:
                            model_turn = server_content.model_turn
                            if model_turn is not None:
                                for part in model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        if not is_model_speaking:
                                            is_model_speaking = True
                                            turn_start_time = loop.time()
                                            sent_bytes_in_turn = 0
                                            logger.info(f" [Gemini Live]: Starting Audio Response Turn...")

                                        pcm_16k = resampler.process(part.inline_data.data)
                                        if pcm_16k:
                                            sent_bytes_in_turn += len(pcm_16k)
                                            await websocket.send_bytes(pcm_16k)

                                    if part.text:
                                        logger.info(f"  [Gemini Live Text]: {part.text.strip()}")

                            if server_content.turn_complete:
                                duration_s = loop.time() - turn_start_time if turn_start_time > 0 else 0
                                leftover_pcm = resampler.flush()
                                if leftover_pcm:
                                    sent_bytes_in_turn += len(leftover_pcm)
                                    await websocket.send_bytes(leftover_pcm)
                                logger.info(f" [Gemini Live]: Turn Complete for session '{session_id}' ({sent_bytes_in_turn} bytes, {duration_s:.2f}s duration)")
                                await websocket.send_text('{"type": "turn_complete"}')
                                is_model_speaking = False
                                turn_start_time = 0.0
                                sent_bytes_in_turn = 0
                except asyncio.CancelledError:
                    pass
                except Exception as rx_err:
                    logger.warning(f"Gemini Live RX loop ended for '{session_id}': {rx_err}")

            rx_task = asyncio.create_task(gemini_rx_loop())



            audio_buffer = bytearray()
            chunk_counter = 0

            try:
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.receive(), timeout=0.1)
                    except asyncio.TimeoutError:
                        if streaming_speech and (loop.time() - last_audio_in_time > 0.85):
                            streaming_speech = False
                            logger.info(f"🎤 [Audio Turn End]: Recorded {len(audio_buffer)} bytes. Triggering Gemini response...")
                            await session.send_realtime_input(audio_stream_end=True)
                            audio_buffer.clear()
                        continue

                    if message["type"] == "websocket.disconnect":
                        return

                    if "bytes" in message and message["bytes"]:
                        last_activity_time = loop.time()
                        last_audio_in_time = loop.time()
                        streaming_speech = True

                        data = message["bytes"]
                        if len(data) == 44 and data.startswith(b"RIFF"):
                            continue
                        pcm_data = data[44:] if data.startswith(b"RIFF") else data
                        if not pcm_data:
                            continue
                        
                        boosted_pcm = boost_chunk_gain(pcm_data)
                        audio_buffer.extend(boosted_pcm)
                        chunk_counter += 1
                        
                        # Send boosted realtime chunk to Live session
                        await session.send_realtime_input(
                            audio=types.Blob(data=boosted_pcm, mime_type="audio/pcm;rate=16000")
                        )
                        
                        if chunk_counter % 25 == 0:
                            logger.info(f" 🎙️ [Audio Stream]: Streaming {len(audio_buffer)} bytes PCM to Gemini...")
                    elif "text" in message and message["text"]:
                        txt = message["text"]
                        if "audio_end" in txt:
                            streaming_speech = False
                            logger.info(f"🎤 [Explicit Turn End]: Triggering Gemini response for session '{session_id}'...")
                            try:
                                await session.send_realtime_input(audio_stream_end=True)
                                await session.send(input=types.Content(role="user", parts=[types.Part.from_text(text="")]))
                            except Exception as err:
                                logger.warning(f"Error sending audio_stream_end: {err}")
                            audio_buffer.clear()
            finally:
                rx_task.cancel()
    except WebSocketDisconnect:
        logger.info(f"🔴 Client disconnected from Gemini Live Stream (session_id: '{session_id}')")
        return
    except Exception as live_err:
        logger.error(f"❌ Gemini Live Exception ({session_id}): {live_err}")
    finally:
        logger.info(f"🔴 Gemini Live Session Closed for session_id='{session_id}'")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f" Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, ws_ping_interval=None, ws_ping_timeout=None, timeout_keep_alive=600)
