# Voice Jitter & Audio Stutter Diagnostic Documentation

## Executive Summary
This document provides a comprehensive technical breakdown of the voice architecture, real-time audio pipeline, FreeRTOS buffering mechanisms, WebSocket network transport, and Python backend streaming for the **ESP32-S3 Gemini Live Voice Assistant**.

This file is structured specifically for LLMs and audio engineering systems to diagnose the root causes behind **sudden high voice jitter, micro-stutters, robotic pitch artifacts, and audio buffer underruns**.

---

## 1. System Architecture Overview

```
┌─────────────────────────┐          WebSocket (TCP)          ┌──────────────────────────┐         Gemini Live API        ┌─────────────────────────┐
│     ESP32-S3 Firmware    │ ◄─────────────────────────────────── │   Python FastAPI Backend │ ◄────────────────────────────► │  Gemini 2.5 Flash Live  │
│  (FreeRTOS + I2S DAC)   │      16kHz 16-bit Mono PCM        │  (Polyphase Resampler)   │    24kHz Bi-directional PCM    │     (Native Audio)      │
└─────────────────────────┘                                   └──────────────────────────┘                                └─────────────────────────┘
```

### End-to-End Pipeline Specifications
- **Microcontroller**: ESP32-S3 (Dual-Core Xtensa LX7 @ 240MHz).
- **Audio Codec**: ES8311 (DAC for Speaker Output) / ES7210 (ADC for Microphone Input).
- **Sample Rate / Format**: 16,000 Hz, 16-bit Signed PCM, Mono (32,000 bytes/sec bandwidth).
- **Network Transport**: WebSocket over TCP (Non-TLS / TLS depending on setup).
- **Backend Relay**: Python FastAPI with `asyncio` loop running `google-genai` Live API connection.
- **Upstream Audio Model**: Gemini Live API outputting 24,000 Hz 16-bit Mono PCM.
- **Resampler**: Studio-grade 24kHz to 16kHz polyphase resampler using 4-point cubic Hermite interpolation.

---

## 2. Firmware Deep-Dive (`main.c` & `bsp_board.c`)

### 2.1 Audio Playback Ringbuffer & FreeRTOS Task Configuration
The ESP32 uses a FreeRTOS RingBuffer (`s_audio_play_rb`) to buffer incoming binary PCM chunks from WebSocket.

```c
// Ringbuffer Creation (main.c)
s_audio_play_rb = xRingbufferCreate(65536, RINGBUF_TYPE_BYTEBUF); // 64 KB capacity (~2.0s audio cushion)

// Dedicated Audio Playback Task (main.c)
xTaskCreatePinnedToCore(audio_playback_task, "audio_play_task", 4096, NULL, 10, NULL, 1);
```

#### Key Task Parameters:
- **Core Affinity**: Pinned to **Core 1** (isolated from Wi-Fi MAC interrupts on Core 0).
- **Task Priority**: Priority **10** (High priority execution).
- **Ringbuffer Type**: `RINGBUF_TYPE_BYTEBUF` (Byte stream without per-item header overhead).
- **Pre-buffer Cushion (`PLAYBACK_PREBUFFER_BYTES`)**: `9600` bytes (~300ms cushion @ 32,000 B/s).

---

### 2.2 Playback Loop Mechanics (`audio_playback_task`)

```c
static void audio_playback_task(void *pvParameters) {
    uint8_t zero_silence[1024] = {0};
    int empty_stall_ms = 0;

    while (1) {
        // Pre-buffering phase to absorb Wi-Fi network jitter
        if (s_is_prebuffering) {
            if (s_buffered_bytes >= PLAYBACK_PREBUFFER_BYTES || (s_turn_complete && s_buffered_bytes > 0)) {
                s_is_prebuffering = false;
            } else {
                vTaskDelay(pdMS_TO_TICKS(15));
                continue;
            }
        }

        size_t item_size = 0;
        uint8_t *item = (uint8_t *)xRingbufferReceive(s_audio_play_rb, &item_size, pdMS_TO_TICKS(100));

        if (item != NULL && item_size > 0) {
            item_size &= ~1; // Align to 16-bit sample boundary

            if (item_size > 0) {
                if (s_buffered_bytes >= item_size) {
                    s_buffered_bytes -= item_size;
                } else {
                    s_buffered_bytes = 0;
                }

                empty_stall_ms = 0;
                s_playback_ctx.is_playing = true;

                if (s_playback_ctx.play_dev) {
                    esp_codec_dev_write(s_playback_ctx.play_dev, (void *)item, item_size);
                }

                s_playback_ctx.total_audio_read += item_size;
            }
            vRingbufferReturnItem(s_audio_play_rb, (void *)item);
        } else {
            // Ringbuffer is momentarily empty
            if (s_playback_ctx.is_playing) {
                empty_stall_ms += 100;

                if (s_turn_complete || empty_stall_ms >= 800) {
                    if (s_playback_ctx.play_dev) {
                        esp_codec_dev_write(s_playback_ctx.play_dev, (void *)zero_silence, sizeof(zero_silence));
                    }
                    s_playback_ctx.is_playing = false;
                    s_turn_complete = true;
                    s_is_prebuffering = true;
                    s_buffered_bytes = 0;
                    empty_stall_ms = 0;
                }
            } else {
                s_is_prebuffering = true;
                s_buffered_bytes = 0;
            }
        }
    }
}
```

---

### 2.3 WebSocket Receiver Handler (`websocket_event_handler`)

```c
static void websocket_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;

    switch (event_id) {
        case WEBSOCKET_EVENT_DATA:
            if (data->op_code == 0x01) { // Text JSON frame
                if (strstr(data->data_ptr, "turn_complete") != NULL) {
                    s_turn_complete = true;
                }
            } else if (data->op_code == 0x02 || data->op_code == 0x00) { // Binary PCM frame
                if (data->data_len > 0 && s_audio_play_rb) {
                    s_turn_complete = false;
                    BaseType_t res = xRingbufferSend(s_audio_play_rb, data->data_ptr, data->data_len, pdMS_TO_TICKS(2500));
                    if (res == pdTRUE) {
                        s_buffered_bytes += data->data_len;
                    } else {
                        ESP_LOGW(TAG, "Audio play ring buffer full, dropped chunk (%d bytes)", data->data_len);
                    }
                }
            }
            break;
    }
}
```

---

## 3. Backend Deep-Dive (`server.py`)

### 3.1 24kHz to 16kHz Resampling Logic
The backend receives 24kHz PCM from Gemini Live API and downsamples to 16kHz before streaming to the ESP32.

```python
class SmoothResampler24kTo16k:
    def __init__(self, volume_scale: float = 0.44):
        self.raw_bytes = bytearray()
        self.history = [0, 0, 0] # 3-sample history prefix
        self.remainder_samples = []
        self.last_out_sample = 0
        self.volume_scale = volume_scale

    def process(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""
        self.raw_bytes.extend(chunk)
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
            
            # Symmetrically matched 4-point cubic Hermite interpolation
            y0_raw = (sm1 + 14 * s0 + s1 + 8) >> 4
            y1_raw = (-s0 + 9 * s1 + 9 * s2 - s3 + 8) >> 4
            
            y0 = max(-32768, min(32767, int(y0_raw * scale)))
            y1 = max(-32768, min(32767, int(y1_raw * scale)))
            
            out_samples.append(y0)
            out_samples.append(y1)
            
        return struct.pack(f"<{len(out_samples)}h", *out_samples)
```

---

### 3.2 Real-Time Flow Control Pacing (`server.py`)
To prevent flooding or starving the ESP32 ringbuffer, the backend calculates lead time and sleeps in Python `asyncio`.

```python
async def gemini_rx_loop():
    nonlocal last_activity_time, is_model_speaking
    sent_bytes_in_turn = 0
    turn_start_time = 0.0

    while True:
        async for response in session.receive():
            server_content = response.server_content
            if server_content is not None and server_content.model_turn:
                for part in server_content.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        pcm_16k = resampler.process(part.inline_data.data)
                        if pcm_16k:
                            if sent_bytes_in_turn == 0:
                                turn_start_time = loop.time()

                            sent_bytes_in_turn += len(pcm_16k)
                            await websocket.send_bytes(pcm_16k)

                            # Real-Time Flow Control (32KB/sec for 16kHz 16-bit Mono)
                            total_audio_sec = sent_bytes_in_turn / 32000.0
                            elapsed_sec = loop.time() - turn_start_time
                            lead_time = total_audio_sec - elapsed_sec
                            if lead_time > 0.60:
                                await asyncio.sleep(lead_time - 0.50)
```

---

## 4. Root Cause Analysis of Sudden High Voice Jitter

The following **5 critical vulnerabilities** are the primary contributors to sudden high jitter, stuttering, and audio dropouts:

### ❌ Vulnerability 1: The 100ms Ringbuffer Receive Timeout Stall (`pdMS_TO_TICKS(100)`)
- **Location**: `main.c` in `audio_playback_task`:
  ```c
  uint8_t *item = (uint8_t *)xRingbufferReceive(s_audio_play_rb, &item_size, pdMS_TO_TICKS(100));
  ```
- **Mechanism**:
  When the ringbuffer briefly drops to 0 bytes due to a minor Wi-Fi jitter pulse (~20–50ms network delay), `xRingbufferReceive()` blocks for **up to 100ms**.
  During this 100ms blocking state, the speaker hardware (`esp_codec_dev_write`) receives **no data**, causing physical DAC audio starvation (instant silence click/pop).
  When network packets arrive 20ms later, the task is still blocked or unblocking, causing chunked playback out-of-sync with real time.

---

### ❌ Vulnerability 2: Python `asyncio.sleep` Timer Resolution & Latency Accumulation
- **Location**: `server.py` inside `gemini_rx_loop`:
  ```python
  total_audio_sec = sent_bytes_in_turn / 32000.0
  elapsed_sec = loop.time() - turn_start_time
  lead_time = total_audio_sec - elapsed_sec
  if lead_time > 0.60:
      await asyncio.sleep(lead_time - 0.50)
  ```
- **Mechanism**:
  1. Default Windows timer resolution for Python `asyncio` is ~15.6ms. `asyncio.sleep()` often over-sleeps by 15–30ms.
  2. The pacing math computes `total_audio_sec` using `sent_bytes_in_turn` accumulated from the start of the turn. Over multi-second turns, slight delays in Gemini API chunk generation cause `elapsed_sec` to grow faster than `total_audio_sec`.
  3. When `lead_time` drops below `0.50`, no sleep occurs and frames burst all at once. When `lead_time` spikes above `0.60`, the loop sleeps for `lead_time - 0.50` seconds (e.g. 200ms+), completely freezing WebSocket transmission and starving the ESP32 ringbuffer!

---

### ❌ Vulnerability 3: Asymmetric Underrun State & The 800ms Stall Window
- **Location**: `main.c` stall handling:
  ```c
  if (s_turn_complete || empty_stall_ms >= 800) { ... }
  ```
- **Mechanism**:
  When the buffer starves mid-sentence, `empty_stall_ms` accumulates in 100ms steps. For **800ms**, the system remains in state `CONV_STATE_SPEAKING` without resetting pre-buffering.
  During this 800ms window, arriving chunks are written immediately item-by-item without re-entering the pre-buffer phase, leading to repetitive micro-stutters ("robotic stutter") until 800ms elapses.

---

### ❌ Vulnerability 4: TCP Packet Nagle Algorithm & Wi-Fi Fragmentation
- **Location**: ESP32 Wi-Fi & WebSocket Client setup (`wifi_connect.c` & `main.c`).
- **Mechanism**:
  By default, TCP sockets use Nagle's algorithm (`TCP_NODELAY` disabled), which buffers small audio chunks (e.g. 512 bytes) on the backend or ESP32 TCP stack before transmitting. This transforms a continuous 32KB/sec stream into bursty 1460-byte MTU arrivals separated by 50–150ms gaps.

---

### ❌ Vulnerability 5: FreeRTOS Task Priority & Socket Receive Preemption
- **Location**: ESP32 WebSocket task vs Audio Playback task priorities.
- **Mechanism**:
  `audio_playback_task` runs at Priority 10 on Core 1. However, `esp_websocket_client` internal RX thread runs at default priority (Priority 5) on Core 0. If Wi-Fi networking routines or background logging preempt the WebSocket RX thread on Core 0, network packets accumulate in socket buffers rather than being pushed to `s_audio_play_rb`.

---

## 5. Recommended Remediation & Code Fixes

### Fix 1: Reduce Ringbuffer Receive Timeout to 5ms in `main.c`
Change `pdMS_TO_TICKS(100)` to `pdMS_TO_TICKS(5)` to prevent long DAC starvation blocks when ringbuffer is empty:
```diff
- uint8_t *item = (uint8_t *)xRingbufferReceive(s_audio_play_rb, &item_size, pdMS_TO_TICKS(100));
+ uint8_t *item = (uint8_t *)xRingbufferReceive(s_audio_play_rb, &item_size, pdMS_TO_TICKS(5));
```

### Fix 2: Smooth Windowed Pacing in `server.py`
Replace absolute cumulative `lead_time` pacing with token bucket or per-chunk micro-pacing (max 10–20ms sleeps) to guarantee steady streaming without long pauses:
```python
# Fixed Pacing Logic in server.py:
target_rate = 32000.0  # bytes per second
chunk_duration = len(pcm_16k) / target_rate
now = loop.time()
next_send_time = max(now, next_send_time + chunk_duration)
sleep_delay = next_send_time - now
if 0.002 < sleep_delay < 0.30:
    await asyncio.sleep(sleep_delay)
```

### Fix 3: Enable TCP_NODELAY on WebSocket Connections
Ensure `TCP_NODELAY` is enabled on both client and server WebSocket sockets to disable packet coalescing.

### Fix 4: Instant Dynamic Re-prebuffering on Buffer Underrun
If ringbuffer hits 0 bytes mid-turn, immediately re-trigger `s_is_prebuffering = true` with a lightweight 100ms cushion (`3200` bytes) instead of waiting for 800ms stall timeout.

---

## Summary Checklist for LLM Diagnosticians
| Area | Parameter | Current Value | Recommended Target | Potential Risk |
|---|---|---|---|---|
| ESP32 | `xRingbufferReceive` Timeout | `100ms` | `5ms` | High (Causes 100ms DAC freezes) |
| ESP32 | `PLAYBACK_PREBUFFER_BYTES` | `9600` bytes (300ms) | `6400` - `9600` bytes | Medium (Network delay absorption) |
| ESP32 | Underrun Stall Timeout | `800ms` | Immediate re-buffer | High (Robotic stutter window) |
| Backend | Pacing Mechanism | Cumulative `lead_time` sleep | Sliding Window / Token Bucket | High (Triggers bursty sleep pauses) |
| Network | TCP Nagle (`TCP_NODELAY`) | Default | Enabled (`true`) | Medium (Packet arrival burstiness) |
