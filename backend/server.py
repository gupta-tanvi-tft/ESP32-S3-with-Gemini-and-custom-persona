import os
import io
import wave
import asyncio
import tempfile
import logging
import struct
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, WebSocket, WebSocketDisconnect
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
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "gemini-2.5-flash",
    "gemini-2.0-flash",
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

    tried_models = set()
    for model_name in FALLBACK_MODELS:
        if model_name in tried_models:
            continue
        tried_models.add(model_name)

        try:
            logger.info(f"Sending {len(audio_bytes)} bytes audio to Gemini Model '{model_name}' for Human Voice & Persona Reasoning...")
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
            logger.warning(f"Model '{model_name}' failed: {e}. Trying next fallback...")
            await asyncio.sleep(0.3)

    logger.error("All Gemini API model attempts failed!")
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
    base_voice = os.getenv("TTS_VOICE", "en-US-AriaNeural")
    
    # Configure dynamic SSML voice parameters based on emotional mood
    if mood == "celebratory":
        rate = "+8%"
        pitch = "+3Hz"
        volume = "+5%"
    elif mood == "calm_reassuring":
        rate = "-8%"
        pitch = "-2Hz"
        volume = "-5%"
    elif mood == "empathetic_gentle":
        rate = "-5%"
        pitch = "-1Hz"
        volume = "-8%"
    else: # warm_clinical
        rate = "+3%"
        pitch = "+0Hz"
        volume = "-10%"

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

# Verified primary model for Live API bidiGenerateContent
LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")

class GeminiLiveTransportSession:
    """
    Manages persistent bi-directional WebSockets transport with Google Gemini 3.1 Live API.
    Handles LiveConnectConfig setup, streaming audio input/output, and interruption events.
    """
    def __init__(self, session_id: str, patient_name: str, persona_str: str):
        self.session_id = session_id
        self.patient_name = patient_name
        self.persona_str = persona_str
        self.live_session = None
        self.is_connected = False

    async def connect(self, api_key: str):
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        system_instruction = (
            "STRICT HUMAN VOICE INTELLIGENCE & CLINICAL ASSISTANT INSTRUCTIONS:\n"
            f"You are a human-like, highly empathetic clinical voice companion speaking directly to {self.patient_name}.\n"
            "Analyze both the spoken content AND the tone of the user's voice.\n"
            "1. IF THE USER SAYS A GREETING OR CASUAL CHITCHAT (e.g., 'Hello', 'Hi', 'Hey', 'Good morning', 'How are you'):\n"
            "   - Greet them back warmly and naturally by name! Ask how they are feeling today.\n"
            "   - DO NOT blurt out clinical data or HbA1c/medical metrics unless specifically asked!\n"
            "2. IF THE USER ASKS A SPECIFIC HEALTH QUESTION:\n"
            "   - Search the PATIENT PERSONA RECORD below and extract exact values (doctor name/ID, HbA1c, glucose, vitals, step count, medications, lab reports, doctor notes, etc.).\n"
            "3. Determine the user's emotion/tone (`happy`, `anxious`, `concerned`, `pain`, `neutral`, `curious`).\n"
            "4. Determine the best voice response mood (`celebratory`, `calm_reassuring`, `empathetic_gentle`, `warm_clinical`).\n"
            "   - Use `celebratory` for positive achievements or friendly greetings.\n"
            "   - Use `calm_reassuring` for user anxiety, elevated glucose spikes, or high blood pressure.\n"
            "   - Use `empathetic_gentle` for pain, discomfort, or missed medication notes.\n"
            "   - Use `warm_clinical` for general informative questions.\n"
            "5. Respond in clear conversational speech using exact data values when asked.\n\n"
            "PATIENT PERSONA RECORD:\n"
            f"{self.persona_str}\n"
        )

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=system_instruction)]
            )
        )

        logger.info(f"⚡ Establishing Gemini Live API Session [{LIVE_MODEL}] for session_id='{self.session_id}'...")
        self.live_session = await client.aio.live.connect(model=LIVE_MODEL, config=config).__aenter__()
        self.is_connected = True
        logger.info(f"✅ Gemini Live API Session ACTIVE for session_id='{self.session_id}'")

    async def send_audio_chunk(self, pcm_bytes: bytes, mime_type: str = "audio/pcm;rate=16000"):
        if self.live_session and self.is_connected:
            from google.genai import types
            await self.live_session.send_realtime_input(
                media=types.Blob(data=pcm_bytes, mime_type=mime_type)
            )

    async def close(self):
        if self.live_session:
            try:
                await self.live_session.close()
            except Exception:
                pass
            self.is_connected = False
            logger.info(f"🔴 Gemini Live Session Closed for session_id='{self.session_id}'")

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
    persona_path = os.path.join(os.path.dirname(__file__), "patient_persona.json")
    persona_str = ""
    patient_name = "Samarth"

    if os.path.exists(persona_path):
        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                persona_str = f.read()
            import json
            p_data = json.loads(persona_str).get("data", {})
            identity = p_data.get("identity", {})
            patient_name = identity.get("first_name", "Samarth")
        except Exception:
            pass

    live_transport = GeminiLiveTransportSession(session_id, patient_name, persona_str)
    try:
        await live_transport.connect(api_key)
    except Exception as err:
        logger.error(f"Failed to connect to Gemini Live API: {err}", exc_info=True)
        await websocket.send_json({"event": "error", "message": "Gemini Live session connection failed."})
        await websocket.close()
        return

    # Task to receive real-time audio output from Gemini Live API and relay to ESP32 client
    async def gemini_rx_loop():
        try:
            async for response in live_transport.live_session.receive():
                server_content = response.server_content
                if server_content is not None:
                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.inline_data:
                                # Relay raw PCM audio frames directly to ESP32 speaker
                                await websocket.send_bytes(part.inline_data.data)
                    if server_content.interrupted:
                        logger.info(f"⚡ [BARGE-IN] Gemini detected user interruption for '{session_id}'! Signaling client...")
                        await websocket.send_json({"event": "interrupted", "session_id": session_id})
        except Exception as rx_err:
            logger.warning(f"Gemini Live RX loop ended for '{session_id}': {rx_err}")

    rx_task = asyncio.create_task(gemini_rx_loop())

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                data = message["bytes"]
                pcm_data = data[44:] if data.startswith(b"RIFF") else data
                await live_transport.send_audio_chunk(pcm_data)
            elif "text" in message and message["text"]:
                if "barge_in" in message["text"] or "stop" in message["text"]:
                    logger.info(f"⚡ Client sent explicit barge-in signal for '{session_id}'")
    except WebSocketDisconnect:
        logger.info(f"🔴 Client disconnected from Gemini Live Stream (session_id: '{session_id}')")
    finally:
        rx_task.cancel()
        await live_transport.close()

@app.websocket("/ws/voice_dynamic/{session_id}")
@app.websocket("/ws/voice_dynamic")
async def websocket_voice_dynamic(websocket: WebSocket, session_id: str = "default"):
    """
    WebSocket WS/WSS Endpoint: /ws/voice_dynamic/{session_id}
    Receives PCM/WAV binary audio frames from ESP32 or web clients over persistent socket.
    Sends back text metadata as JSON and 16kHz 16-bit Mono PCM audio bytes to the speaker.
    """
    await websocket.accept()
    logger.info(f"🟢 [STATE: READY] WebSocket connected (session_id: '{session_id}'). Client can SPEAK now.")

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                logger.info(f"🔴 [STATE: DISCONNECTED] WebSocket client disconnected (session_id: '{session_id}').")
                break

            if "bytes" in message and message["bytes"]:
                data = message["bytes"]
            elif "text" in message and message["text"]:
                logger.info(f"[{session_id}] Received text WS frame: {message['text']}")
                continue
            else:
                continue

            # If client sends a 44-byte WAV header first frame, receive next frame for PCM body
            if len(data) == 44 and data.startswith(b"RIFF"):
                logger.info(f"[{session_id}] Received WAV header frame, waiting for PCM payload frame...")
                next_msg = await websocket.receive()
                if "bytes" in next_msg and next_msg["bytes"]:
                    data = data + next_msg["bytes"]

            if len(data) < 100:
                logger.warning(f"[{session_id}] Received audio payload too small ({len(data)} bytes), skipping...")
                continue

            logger.info(f" [STATE: RECORDED] [{session_id}] Received {len(data)} bytes audio. DO NOT SPEAK - Processing with Gemini...")

            # Extract PCM payload if header is WAV (RIFF)
            pcm_payload = data[44:] if data.startswith(b"RIFF") else data


            # 3. Audio gain normalization & WAV wrapping for Gemini
            processed_pcm = process_audio_pcm(pcm_payload)
            wav_bytes = pcm_to_wav(processed_pcm, sample_rate=16000, channels=1, sample_width=2)

            # 4. Call Gemini Model API (Perceives user emotion & derives persona answer)
            logger.info(f"🧠 [STATE: THINKING] Processing human voice & persona for [{session_id}]...")
            gemini_res = await call_gemini_api(wav_bytes, mime_type="audio/wav")

            answer_text = gemini_res.get("answer", "I'm here to help you.")
            response_mood = gemini_res.get("response_mood", "warm_clinical")
            user_emotion = gemini_res.get("user_emotion", "neutral")

            # 5. Synthesize mood-modulated SSML TTS to 16kHz 16-bit Mono PCM
            logger.info(f"🔊 [STATE: SYNTHESIZING] Converting response to '{response_mood}' SSML speech...")
            pcm_audio_output = await text_to_pcm_16k(answer_text, mood=response_mood)

            # 6. Detect volume control commands
            lower_text = answer_text.lower()
            vol_command = None
            if "mute" in lower_text or "silent" in lower_text:
                vol_command = 0
            elif "increase volume" in lower_text or "volume up" in lower_text or "louder" in lower_text:
                vol_command = 85
            elif "lower volume" in lower_text or "volume down" in lower_text or "softer" in lower_text or "quiet" in lower_text:
                vol_command = 35

            # 7. Send JSON metadata frame first
            meta_payload = {
                "event": "response",
                "state": "speaking",
                "session_id": session_id,
                "text": answer_text,
                "user_emotion": user_emotion,
                "response_mood": response_mood,
                "audio_bytes": len(pcm_audio_output)
            }
            if vol_command is not None:
                meta_payload["set_volume"] = vol_command

            await websocket.send_json(meta_payload)

            # 8. Send raw PCM audio binary frame to ESP32 speaker
            logger.info(f"📢 [STATE: SPEAKING] [{session_id}] Streaming {len(pcm_audio_output)} bytes ({response_mood}) audio to speaker.")
            await websocket.send_bytes(pcm_audio_output)
            logger.info(f"🟢 [STATE: READY] [{session_id}] Finished playing response. Client can SPEAK now.")


    except WebSocketDisconnect:
        logger.info(f" WebSocket client disconnected (session_id: '{session_id}')")
    except Exception as e:
        logger.error(f" [{session_id}] WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8008))
    logger.info(f" Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
