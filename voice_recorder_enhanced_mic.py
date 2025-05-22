import streamlit as st
import sounddevice as sd
import numpy as np
import time
import os
import wave
import assemblyai as aai
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Configuration
SAMPLE_RATE = 44100  # Sample rate (Hz)
CHANNELS = 1         # Mono audio
RECORDINGS_DIR = "recordings"
COMBINED_DIR = "combined"
TAG_OPTIONS = ["💖 Personal", "❓ Question", "⚡ Priority", "😎 Chill"]

# Ensure directories exist
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

# Initialize session state defaults
defaults = {
    'is_recording': False,
    'recording_start_time': 0,
    'recordings': [],
    'input_volume': 1.0,
    'api_key': os.getenv('ASSEMBLY_API_KEY', ''),
    'debug': []
}
for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default
if 'audio_frames' not in st.session_state:
    st.session_state['audio_frames'] = []
# allow users to pick which recordings to combine
if 'combine_selection' not in st.session_state:
    st.session_state.combine_selection = set()

# Debug logging


def add_debug(msg):
    st.session_state.debug.append(f"{time.strftime('%H:%M:%S')}: {msg}")

# Transcribe audio via AssemblyAI


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

# Combine recordings into one WAV


def combine_audio_files():
    selected_indices = st.session_state.combine_selection
    if not selected_indices:
        add_debug("No recordings selected for combining")
        return None

    segments = []
    for idx in selected_indices:
        rec = st.session_state.recordings[idx]
        try:
            with wave.open(rec['filepath'], 'rb') as wf:
                data = wf.readframes(wf.getnframes())
                arr = np.frombuffer(data, dtype=np.int16)
                segments.append(arr)
        except Exception as e:
            add_debug(f"Error reading {rec['filepath']}: {e}")

    if not segments:
        return None

    combined = np.concatenate(segments)
    filename = f"combined_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(COMBINED_DIR, filename)
    with wave.open(out_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(combined.tobytes())
    add_debug(f"Combined WAV saved: {out_path}")
    return out_path

# Callback for tag changes


def on_tag_change(idx):
    new = st.session_state[f"tag_{idx}"]
    st.session_state.recordings[idx]['tag'] = new
    add_debug(f"Recording {idx+1} tagged as {new}")


# App UI
st.title("🎙️ Voice Recorder with Tags")

# Audio Settings
with st.expander("Audio Settings", expanded=True):
    st.session_state.input_volume = st.slider(
        "Input Volume Multiplier", 0.1, 5.0,
        value=st.session_state.input_volume, step=0.1
    )

# API Key Input
with st.expander("API Settings", expanded=False):
    key = st.text_input(
        "AssemblyAI API Key", value=st.session_state.api_key, type='password'
    )
    if key != st.session_state.api_key:
        st.session_state.api_key = key
        add_debug("API key updated")

# Recording Controls


def record_controls():
    frames = st.session_state.audio_frames
    col1, col2 = st.columns(2)
    if not st.session_state.is_recording:
        if col1.button("Start Recording", use_container_width=True):
            frames.clear()

            def callback(indata, _, __, status):
                if status:
                    add_debug(f"Stream status: {status}")
                frames.append(indata.copy())
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                callback=callback
            )
            stream.start()
            st.session_state.stream = stream
            st.session_state.is_recording = True
            st.session_state.recording_start_time = time.time()
            add_debug("Recording started")
    else:
        if col2.button("Stop Recording", use_container_width=True):
            stream = st.session_state.get('stream')
            if stream:
                stream.stop()
                stream.close()
            st.session_state.is_recording = False
            add_debug("Recording stopped")
            if not frames:
                st.error("No audio captured. Check mic or close other apps.")
                add_debug("audio_frames empty")
            else:
                audio = np.concatenate(frames, axis=0).flatten()
                audio = np.clip(
                    audio * st.session_state.input_volume, -1.0, 1.0)
                data16 = (audio * 32767).astype(np.int16)
                fname = f"recording_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
                fpath = os.path.join(RECORDINGS_DIR, fname)
                with wave.open(fpath, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(data16.tobytes())
                add_debug(f"Saved WAV: {fpath}")
                st.audio(fpath)
                with st.spinner("Transcribing..."):
                    txt = transcribe_audio(fpath, st.session_state.api_key)
                st.success("Transcription complete 🎉")
                st.write(f"📝 {txt}")
                st.session_state.recordings.append({
                    'filepath': fpath,
                    'duration': len(audio)/SAMPLE_RATE,
                    'text': txt,
                    'tag': TAG_OPTIONS[0]
                })


with st.container():
    record_controls()

# Combine Selected Recordings Button
selected_count = len(st.session_state.combine_selection)
button_text = f"Combine Selected ({selected_count})" if selected_count > 0 else "Select recordings to combine"
if st.button(button_text, use_container_width=True, disabled=selected_count == 0):
    combined_path = combine_audio_files()
    if combined_path:
        st.success(f"Combined audio saved: {combined_path}")
        st.audio(combined_path)
    else:
        st.warning("Failed to combine recordings")

# Live status
if st.session_state.is_recording:
    elapsed = time.time() - st.session_state.recording_start_time
    st.write(f"🔴 Recording... {elapsed:.1f}s")

# List recordings with tag selector
st.markdown("---")
total = len(st.session_state.recordings)
for rev_idx, rec in enumerate(reversed(st.session_state.recordings)):
    orig_idx = total - 1 - rev_idx

    # Add selection checkbox
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.checkbox("", key=f"select_{orig_idx}"):
            st.session_state.combine_selection.add(orig_idx)
        else:
            st.session_state.combine_selection.discard(orig_idx)

    with col2:
        st.subheader(f"Recording {orig_idx+1}")
        # Show tag badge
        badge = rec.get('tag', TAG_OPTIONS[0])
        st.markdown(
            f"<span style='background-color:#444;color:#fff;padding:4px 8px;border-radius:4px;'>{badge}</span>", unsafe_allow_html=True)
        # Audio + details
        st.audio(rec['filepath'])
        st.write(f"⏱️ Duration: {rec.get('duration', 0):.2f}s")
        st.write(f"📝 {rec.get('text', '')}")
        # Tag dropdown with on_change callback
        st.selectbox(
            "Change Tag", TAG_OPTIONS,
            index=TAG_OPTIONS.index(badge),
            key=f"tag_{orig_idx}",
            on_change=on_tag_change,
            args=(orig_idx,)
        )

# Debug log
with st.expander("Debug Log", expanded=False):
    for msg in st.session_state.debug:
        st.write(msg)
