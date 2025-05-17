import streamlit as st
import sounddevice as sd
import numpy as np
import time
import os
import wave
import assemblyai as aai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
TEMP_AUDIO_FILE = "recording.wav"

def record_audio(duration=5):
    """Record audio for a specified duration in seconds"""
    st.write(f"Recording for {duration} seconds...")
    
    # Record audio
    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16'
    )
    
    # Wait for the recording to complete
    sd.wait()
    
    st.write("Recording complete!")
    return audio_data

def save_audio(audio_data, filename=TEMP_AUDIO_FILE):
    """Save audio data to a WAV file"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio (2 bytes)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    
    # Verify the file was created
    file_size = os.path.getsize(filename)
    st.write(f"Audio saved to {filename} (Size: {file_size} bytes)")
    
    return filename

def transcribe_audio(filename, api_key):
    """Transcribe audio file using AssemblyAI"""
    st.write(f"Setting API key: {api_key[:5]}...{api_key[-4:]}")
    aai.settings.api_key = api_key
    
    st.write("Creating transcriber...")
    transcriber = aai.Transcriber()
    
    st.write(f"Sending file {filename} to AssemblyAI...")
    transcript = transcriber.transcribe(filename)
    
    st.write(f"Transcription complete. Status: {transcript.status}")
    
    # Extract words with timestamps if available
    words = []
    if hasattr(transcript, 'words') and transcript.words:
        st.write(f"Found {len(transcript.words)} words with timestamps")
        words = [{
            'text': word.text,
            'start': word.start / 1000.0,  # Convert ms to seconds
            'end': word.end / 1000.0       # Convert ms to seconds
        } for word in transcript.words]
    
    return {
        'text': transcript.text,
        'words': words
    }

# Streamlit UI
st.title("Simple Voice Recorder with AssemblyAI")

# Initialize session state
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []

# API key input
api_key = st.text_input(
    "Enter your AssemblyAI API key", 
    value=os.getenv("ASSEMBLY_API_KEY", ""), 
    type="password"
)

if api_key:
    os.environ["ASSEMBLY_API_KEY"] = api_key
    st.write("API key stored in environment variable")

# Recording duration
duration = st.slider("Recording Duration (seconds)", 5, 60, 10)

# Record button
if st.button("Record and Transcribe"):
    if not api_key:
        st.error("Please enter your AssemblyAI API key first")
    else:
        try:
            # Record audio
            with st.spinner("Recording..."):
                audio_data = record_audio(duration)
            
            # Save the audio
            with st.spinner("Saving audio..."):
                audio_file = save_audio(audio_data)
            
            # Display audio player
            st.audio(audio_file)
            
            # Transcribe
            with st.spinner("Transcribing audio with AssemblyAI..."):
                st.write("Starting transcription process...")
                result = transcribe_audio(audio_file, api_key)
                
                # Add timestamp for display
                result['timestamp'] = time.strftime("%H:%M:%S")
                
                # Add to session state
                st.session_state.transcriptions.append(result)
                
            # Success message
            st.success("Transcription complete!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Display transcriptions
if st.session_state.transcriptions:
    st.subheader("Transcriptions")
    
    for idx, t in enumerate(st.session_state.transcriptions):
        with st.expander(f"Recording {idx+1} - {t['timestamp']}"):
            st.write("Transcribed text:")
            st.write(t['text'])
            
            # Display words with timestamps if available
            if t['words']:
                st.subheader("Words with Timestamps")
                for word in t['words']:
                    st.write(f"[{word['start']:.2f}s - {word['end']:.2f}s] {word['text']}")
