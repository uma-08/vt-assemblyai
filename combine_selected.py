import streamlit as st
import sounddevice as sd
import numpy as np
import time
import os
import wave
import assemblyai as aai
from dotenv import load_dotenv
import uuid

# Load env
load_dotenv()

# Audio config
SAMPLE_RATE = 44100
CHANNELS = 1
RECORDINGS_DIR = "recordings"
COMBINED_DIR = "combined"
TAG_OPTIONS = ["💖 Personal", "❓ Question", "⚡ Priority", "😎 Chill"]

# Ensure dirs
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

# Session state defaults
defaults = {
    'is_recording': False,
    'start_time': 0,
    'recordings': [],
    'input_volume': 1.0,
    'api_key': os.getenv('ASSEMBLY_API_KEY', ''),
    'combine_selection': set(),
    'selected_device': None,
    'debug': []
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# Debug helper


def add_debug(msg):
    st.session_state.debug.append(f"{time.strftime('%H:%M:%S')}: {msg}")

# Simple blocking recorder


def record_audio(duration, device, volume):
    data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                  channels=CHANNELS, dtype='float32', device=device)
    sd.wait()
    flat = data.flatten()
    flat = np.clip(flat * volume, -1.0, 1.0)
    return (flat * 32767).astype(np.int16)

# Transcription


def transcribe(path, key):
    if not key:
        return ''
    aai.settings.api_key = key
    trans = aai.Transcriber().transcribe(path)
    return getattr(trans, 'text', '') or ''

# Combine selected


def combine_selected():
    sel = sorted(st.session_state.combine_selection)
    if not sel:
        st.warning("No recordings selected to combine.")
        return
    segments = []
    for i in sel:
        rec = st.session_state.recordings[i]
        with wave.open(rec['path'], 'rb') as wf:
            data = wf.readframes(wf.getnframes())
            segments.append(np.frombuffer(data, dtype=np.int16))
    combo = np.concatenate(segments)
    fn = f"combined_{uuid.uuid4().hex[:8]}.wav"
    out = os.path.join(COMBINED_DIR, fn)
    with wave.open(out, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(combo.tobytes())
    st.success(f"Combined saved: {out}")
    st.audio(out)


# UI
st.title("🎙️ Voice Recorder with Tags & Selective Combine")

# Settings panel
with st.expander("Audio Settings", True):
    # Device list
    devs = sd.query_devices()
    inputs = [None] + \
        [i for i, d in enumerate(devs) if d['max_input_channels'] > 0]
    names = ["Default"] + [d['name']
                           for d in devs if d['max_input_channels'] > 0]
    choice = st.selectbox("Microphone", names, index=0)
    st.session_state.selected_device = inputs[names.index(choice)]
    st.session_state.input_volume = st.slider("Volume Multiplier", 0.1, 5.0,
                                              value=st.session_state.input_volume,
                                              step=0.1)
with st.expander("API Settings", False):
    st.session_state.api_key = st.text_input("AssemblyAI API Key",
                                             value=st.session_state.api_key,
                                             type='password')

# Record controls
col1, col2 = st.columns(2)
if not st.session_state.is_recording:
    if col1.button("Start Recording"):
        st.session_state.is_recording = True
        st.session_state.start_time = time.time()
        add_debug("Recording started")
else:
    if col2.button("Stop Recording"):
        st.session_state.is_recording = False
        dur = time.time() - st.session_state.start_time
        add_debug(f"Recording stopped ({dur:.2f}s)")
        audio16 = record_audio(dur,
                               st.session_state.selected_device,
                               st.session_state.input_volume)
        fn = f"rec_{uuid.uuid4().hex[:8]}.wav"
        path = os.path.join(RECORDINGS_DIR, fn)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio16.tobytes())
        # Playback & transcribe
        st.audio(path)
        txt = transcribe(path, st.session_state.api_key)
        st.write(f"📝 {txt}")
        # Store
        st.session_state.recordings.append({
            'path': path,
            'dur': dur,
            'text': txt,
            'tag': TAG_OPTIONS[0]
        })

# List recordings
st.markdown("---")
for i, rec in enumerate(st.session_state.recordings):
    st.subheader(f"Recording {i+1}")
    # checkbox
    sel = st.checkbox("Select to combine", key=f"select_{i}",
                      value=(i in st.session_state.combine_selection))
    if sel:
        st.session_state.combine_selection.add(i)
    else:
        st.session_state.combine_selection.discard(i)
    # tag badge
    badge = rec['tag']
    st.markdown(f"<span style='background:#333;color:#fff;"
                f"padding:4px 8px;border-radius:4px;'>{badge}</span>",
                unsafe_allow_html=True)
    st.audio(rec['path'])
    st.write(f"⏱️ {rec['dur']:.2f}s")
    st.write(f"📝 {rec['text']}")
    # change tag
    new = st.selectbox("Change Tag", TAG_OPTIONS,
                       index=TAG_OPTIONS.index(badge),
                       key=f"tag_{i}")
    if new != badge:
        st.session_state.recordings[i]['tag'] = new
        add_debug(f"Recording {i+1} tagged {new}")

# Combine button
if st.button("Combine Selected Recordings"):
    combine_selected()

# Debug log
with st.expander("Debug Log", False):
    for msg in st.session_state.debug:
        st.write(msg)

