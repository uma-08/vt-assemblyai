import streamlit as st
import sounddevice as sd
import numpy as np
import time
import os
import wave
import assemblyai as aai
from dotenv import load_dotenv
import uuid


def record_audio_simple(duration, device, volume):
    # Blocking record for elapsed time
    data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                  channels=CHANNELS, dtype='float32', device=device)
    sd.wait()
    flat = data.flatten()
    flat = np.clip(flat * volume, -1.0, 1.0)
    return (flat * 32767).astype(np.int16)


# Constants
defaults = {'is_recording': False, 'start_time': 0,
            'recordings': [], 'input_volume': 1.0,
            'api_key': '', 'combine_selection': set(),
            'selected_device': None}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# UI Start
st.title("🎙️ Voice Recorder")

# Settings
devices = [(None, 'Default')] + [(i, d['name'])
                                 for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
sel = st.selectbox("Mic", [n for _, n in devices])
st.session_state.selected_device = devices[[
    n for _, n in devices].index(sel)][0]
st.session_state.input_volume = st.slider(
    "Volume", 0.1, 5.0, st.session_state.input_volume)
st.session_state.api_key = st.text_input(
    "API Key", type='password', value=st.session_state.api_key)

# Recording buttons
col1, col2 = st.columns(2)
if not st.session_state.is_recording and col1.button("Start Recording"):
    st.session_state.is_recording = True
    st.session_state.start_time = time.time()
elif st.session_state.is_recording and col2.button("Stop Recording"):
    st.session_state.is_recording = False
    dur = time.time()-st.session_state.start_time
    audio16 = record_audio_simple(
        dur, st.session_state.selected_device, st.session_state.input_volume)
    fn = f"rec_{uuid.uuid4().hex}.wav"
    path = os.path.join(RECORDINGS_DIR, fn)  # noqa: F821
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(CHANNELS)  # noqa: F821
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)  # noqa: F821
        wf.writeframes(audio16.tobytes())
    st.audio(path)
    # Transcribe
    aai.settings.api_key = st.session_state.api_key
    res = aai.Transcriber().transcribe(path).text
    st.write(res)
    st.session_state.recordings.append(
        {'path': path, 'dur': dur, 'text': res, 'tag': '💖 Personal'})

# List recordings...
# Checkboxes for combine, tags dropdown, combine button
# ... (rest remains similar)

# Combine selected recordings

# Debug
st.write(st.session_state)
