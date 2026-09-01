import asyncio
import miniaudio
import edge_tts

async def main():
    communicate = edge_tts.Communicate("Hello! This is a test of Gemini voice speaker.", "en-US-AvaNeural")
    await communicate.save("test.mp3")
    decoded = miniaudio.decode_file("test.mp3")
    print(f"Decoded MP3: {decoded.sample_rate}Hz, {decoded.nchannels}ch, format={decoded.sample_format}")
    raw_pcm = miniaudio.convert_frames(
        decoded.sample_format, decoded.nchannels, decoded.sample_rate, bytes(decoded.samples),
        miniaudio.SampleFormat.SIGNED16, 1, 16000
    )
    print(f"✅ Converted PCM: 16000Hz 1ch 16-bit, size = {len(raw_pcm)} bytes")

asyncio.run(main())
