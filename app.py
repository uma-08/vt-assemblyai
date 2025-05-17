import streamlit as st
import threading
import time
from audio_processor import (
    callback, recorder, process_audio_chunk,
    CHUNK_DURATION, SAMPLE_RATE,
    CHANNELS, CHUNK_SIZE
)
import sounddevice as sd
import os
import queue
import numpy as np

# Initialize session state variables
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'buffer' not in st.session_state:
    st.session_state.buffer = []
if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'record_thread' not in st.session_state:
    st.session_state.record_thread = None
if 'q' not in st.session_state:
    st.session_state.q = queue.Queue()


def start_recording():
    """Start the audio recording process"""
    if not st.session_state.recording:
        st.session_state.recording = True
        st.session_state.buffer = []  # Clear buffer on new recording
        st.session_state.stream = sd.InputStream(
            callback=callback,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE
        )
        st.session_state.stream.start()
        st.session_state.record_thread = threading.Thread(
            target=recorder,
            daemon=True
        )
        st.session_state.record_thread.start()


def stop_recording():
    """Stop the audio recording process"""
    if st.session_state.recording:
        st.session_state.recording = False
        if st.session_state.stream:
            st.session_state.stream.stop()
            st.session_state.stream.close()
        st.session_state.stream = None
        st.session_state.record_thread = None


def display_transcription():
    """Display the current transcription buffer"""
    if not st.session_state.buffer:
        st.warning("No audio segments transcribed yet!")
        return

    # Get the time window from the slider
    time_window = st.session_state.time_window * 60  # Convert minutes to seconds

    # Display results
    st.subheader("Transcribed Content")

    for i, seg in enumerate(st.session_state.buffer, 1):
        st.write(f"[{seg['start']:.2f}s] {seg['text']}")


# Streamlit UI
st.title("Real-Time Voice Recording & Transcription")

# API key input
assembly_api_key = st.text_input(
    "Enter your AssemblyAI API Key", type="password")
if assembly_api_key:
    os.environ["ASSEMBLY_API_KEY"] = assembly_api_key

# Recording controls
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Recording", disabled=st.session_state.recording):
        if not assembly_api_key:
            st.error("Please enter your AssemblyAI API Key first!")
        else:
            start_recording()
with col2:
    if st.button("Stop Recording", disabled=not st.session_state.recording):
        stop_recording()

# Recording status
if st.session_state.recording:
    st.success("Recording in progress...")
else:
    st.info("Recording stopped")

# Time window slider for display
st.slider(
    "Display segments by time window (minutes)",
    min_value=1,
    max_value=30,
    value=5,
    key="time_window"
)

# Display button
if st.button("Display Transcription"):
    display_transcription()

# Display current buffer
if st.session_state.buffer:
    st.subheader("Current Transcription")
    for i, seg in enumerate(st.session_state.buffer):
        st.write(f"{i+1}. [{seg['start']:.2f}s] {seg['text']}")

# Auto-refresh the page every 5 seconds when recording
if st.session_state.recording:
    time.sleep(5)
    st.experimental_rerun()
