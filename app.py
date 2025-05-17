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
if 'error' not in st.session_state:
    st.session_state.error = None

def start_recording():
    """Start the audio recording process"""
    try:
        if not st.session_state.recording:
            st.session_state.recording = True
            st.session_state.buffer = []  # Clear buffer on new recording
            st.session_state.error = None
            
            # Initialize the queue
            global q
            q = queue.Queue()
            st.session_state.q = q
            
            # Start the stream
            st.session_state.stream = sd.InputStream(
                callback=callback,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE
            )
            st.session_state.stream.start()
            
            # Start recording thread
            def safe_recorder():
                try:
                    recorder()
                except Exception as e:
                    st.session_state.error = f"Recording error: {str(e)}"
                    st.session_state.recording = False
            
            st.session_state.record_thread = threading.Thread(
                target=safe_recorder,
                daemon=True
            )
            st.session_state.record_thread.start()
            
    except Exception as e:
        st.session_state.error = f"Failed to start recording: {str(e)}"
        st.session_state.recording = False

def stop_recording():
    """Stop the audio recording process"""
    try:
        if st.session_state.recording:
            st.session_state.recording = False
            if st.session_state.stream:
                st.session_state.stream.stop()
                st.session_state.stream.close()
            st.session_state.stream = None
            st.session_state.record_thread = None
    except Exception as e:
        st.session_state.error = f"Error stopping recording: {str(e)}"

def display_transcription():
    """Display the current transcription buffer"""
    if not st.session_state.buffer:
        st.warning("No audio segments transcribed yet!")
        return
    
    # Display results
    st.subheader("Transcribed Content")
    
    for i, seg in enumerate(st.session_state.buffer, 1):
        st.write(f"[{seg['start']:.2f}s] {seg['text']}")

# Streamlit UI
st.title("Real-Time Voice Recording & Transcription")

# API key input
assembly_api_key = st.text_input("Enter your AssemblyAI API Key", type="password")
if assembly_api_key:
    os.environ["ASSEMBLY_API_KEY"] = assembly_api_key

# Recording controls
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Recording", disabled=st.session_state.recording, key="start_btn"):
        if not assembly_api_key:
            st.error("Please enter your AssemblyAI API Key first!")
        else:
            start_recording()
with col2:
    if st.button("Stop Recording", disabled=not st.session_state.recording, key="stop_btn"):
        stop_recording()

# Display error if any
if st.session_state.error:
    st.error(st.session_state.error)
    st.session_state.error = None  # Clear after displaying

# Recording status
if st.session_state.recording:
    st.success("Recording in progress...")
else:
    st.info("Recording stopped")

# Display current buffer
if st.session_state.buffer:
    st.subheader("Current Transcription")
    for i, seg in enumerate(st.session_state.buffer):
        st.write(f"{i+1}. [{seg['start']:.2f}s] {seg['text']}")
