import streamlit as st
import sounddevice as sd
import numpy as np
import time
import os
import wave
import assemblyai as aai
from dotenv import load_dotenv
import uuid
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Configuration
SAMPLE_RATE = 44100  # Sample rate (Hz)
CHANNELS = 1         # Mono
RECORDINGS_DIR = "recordings"
COMBINED_DIR = "combined"

# Ensure directories
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

# Initialize session state defaults
for key, default in {
    'is_recording': False,
    'recording_start_time': 0,
    'recordings': [],
    'selected_device': None,
    'input_volume': 1.0,
    'api_key': os.getenv('ASSEMBLY_API_KEY', ''),
    'debug': []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
# Persistent audio frame buffer
tt = 'audio_frames'
if tt not in st.session_state:
    st.session_state[tt] = []

# Debug helper


def add_debug(msg):
    st.session_state.debug.append(f"{time.strftime('%H:%M:%S')}: {msg}")

# List input devices


def get_audio_devices():
    try:
        devices = sd.query_devices()
        inputs = []
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                inputs.append(
                    {'index': idx, 'name': dev['name'], 'default': dev.get('default_input', False)})
        add_debug(f"Found {len(inputs)} input devices")
        return inputs
    except Exception as e:
        add_debug(f"Error querying devices: {e}")
        return []

# Transcribe with AssemblyAI


def transcribe_audio(path, api_key):
    if not api_key:
        add_debug("No API key for transcription")
        return ''
    aai.settings.api_key = api_key
    add_debug(f"Transcribing {path}")
    transcriber = aai.Transcriber()
    result = transcriber.transcribe(path)
    text = getattr(result, 'text', '') or 'No transcription'
    add_debug(f"Received transcription ({len(text)} chars)")
    return text


# App UI
st.title("🎙️ Voice Recorder")

# Audio Settings
with st.expander("Audio Settings", expanded=True):
    devices = get_audio_devices()
    options = ["Default"] + [d['name'] for d in devices]
    default_idx = next((i+1 for i, d in enumerate(devices) if d['default']), 0)
    sel = st.selectbox("Select Microphone", options, index=default_idx)
    if sel != "Default":
        st.session_state.selected_device = devices[options.index(
            sel) - 1]['index']
    else:
        st.session_state.selected_device = None
    st.session_state.input_volume = st.slider(
        "Input Volume Multiplier", 0.1, 5.0,
        value=st.session_state.input_volume,
        step=0.1
    )

# API Key Input
with st.expander("API Settings", expanded=False):
    key = st.text_input("AssemblyAI API Key",
                        value=st.session_state.api_key, type='password')
    if key != st.session_state.api_key:
        st.session_state.api_key = key
        add_debug("API key updated")

# Recording Controls
dev = st.session_state.selected_device
vol = st.session_state.input_volume
frames_list = st.session_state.audio_frames
col1, col2 = st.columns(2)
if not st.session_state.is_recording:
    if col1.button("Start Recording", use_container_width=True):
        # Clear buffer
        frames_list.clear()
        # Define callback closure

        def callback(indata, frames, time_info, status):
            if status:
                add_debug(f"Stream status: {status}")
            frames_list.append(indata.copy())
        # Start stream
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                device=dev,
                dtype='float32',
                callback=callback
            )
            stream.start()
            st.session_state.stream = stream
            st.session_state.is_recording = True
            st.session_state.recording_start_time = time.time()
            add_debug(f"Recording started on device {dev}")
        except Exception as e:
            st.error(f"Failed to start recording: {e}")
            add_debug(f"Stream start error: {e}")
else:
    if col2.button("Stop Recording", use_container_width=True):
        # Stop stream
        stream = st.session_state.get('stream')
        if stream:
            stream.stop()
            stream.close()
        st.session_state.is_recording = False
        add_debug("Recording stopped")
        # Process frames
        if not frames_list:
            st.error("No audio captured. Check mic or close other apps using it.")
            add_debug("audio_frames empty")
        else:
            flat = np.concatenate(frames_list, axis=0).flatten()
            flat = np.clip(flat * vol, -1.0, 1.0)
            int16 = (flat * 32767).astype(np.int16)
            fname = f"recording_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
            fpath = os.path.join(RECORDINGS_DIR, fname)
            with wave.open(fpath, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(int16.tobytes())
            add_debug(f"Saved WAV: {fpath}")
            # Playback
            st.audio(fpath)
            # Transcribe with spinner
            with st.spinner("Transcribing..."):
                txt = transcribe_audio(fpath, st.session_state.api_key)
            st.success("Transcription complete 🎉")
            st.write(f"📝 {txt}")
            # Save record
            st.session_state.recordings.append({
                'filepath': fpath,
                'duration': len(flat)/SAMPLE_RATE,
                'text': txt
            })

# Live status
if st.session_state.is_recording:
    elapsed = time.time() - st.session_state.recording_start_time
    st.write(f"🔴 Recording... {elapsed:.1f}s")

# List recordings
st.markdown("---")
for i, rec in enumerate(reversed(st.session_state.recordings), 1):
    st.subheader(f"Recording {i}")
    st.audio(rec['filepath'])
    st.write(f"⏱️ Duration: {rec.get('duration', 0):.2f}s")
    st.write(f"📝 {rec.get('text', '')}")

# Debug log
with st.expander("Debug Log", expanded=False):
    for msg in st.session_state.debug:
        st.write(msg)
