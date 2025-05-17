import sounddevice as sd
import numpy as np
import queue
import threading
import time
import wave
import requests
import json
import os
from typing import List, Dict
from datetime import datetime

# Global variables
q = queue.Queue()
buffer: List[Dict] = []
CHUNK_DURATION = 10  # seconds (reduced from 30 for faster feedback)
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

def callback(indata, frames, time, status):
    """Callback function for audio stream"""
    if status:
        print(f"Status: {status}")
    q.put(indata.copy())

def save_audio_chunk(chunk: np.ndarray, filename: str):
    """Save audio chunk to WAV file"""
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 2 bytes per sample
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(chunk.tobytes())
    wf.close()

def process_audio_chunk(filename: str):
    """Process audio chunk using AssemblyAI"""
    try:
        api_key = os.getenv("ASSEMBLY_API_KEY")
        if not api_key:
            print("Please set ASSEMBLY_API_KEY environment variable")
            return
        
        # Upload the audio file to AssemblyAI
        headers = {
            "authorization": api_key
        }
        
        # Step 1: Upload the audio file
        print(f"Uploading audio file: {filename}")
        with open(filename, "rb") as f:
            response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers,
                data=f
            )
        
        if response.status_code != 200:
            print(f"Error uploading audio: {response.text}")
            return
        
        upload_url = response.json().get("upload_url")
        if not upload_url:
            print(f"No upload URL in response: {response.text}")
            return
            
        print("Audio uploaded successfully")
        
        # Step 2: Submit the transcription request
        transcription_request = {
            "audio_url": upload_url,
            "punctuate": True,
            "format_text": True,
            "word_timestamps": True    # Get timestamps for each word
        }
        
        print("Submitting transcription request")
        response = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            json=transcription_request,
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"Error submitting transcription request: {response.text}")
            return
        
        transcript_id = response.json().get("id")
        if not transcript_id:
            print(f"No transcript ID in response: {response.text}")
            return
            
        print(f"Transcription request submitted with ID: {transcript_id}")
        
        # Step 3: Poll for the transcription result
        polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
        
        max_retries = 30  # Set a reasonable limit for polling attempts
        retry_count = 0
        
        while retry_count < max_retries:
            print(f"Polling for results (attempt {retry_count+1})")
            polling_response = requests.get(polling_endpoint, headers=headers)
            
            if polling_response.status_code != 200:
                print(f"Error polling for results: {polling_response.text}")
                retry_count += 1
                time.sleep(2)
                continue
                
            polling_response_json = polling_response.json()
            status = polling_response_json.get("status")
            
            if status == "completed":
                print("Transcription completed successfully")
                # Process the completed transcription
                if "words" in polling_response_json and polling_response_json["words"]:
                    words = polling_response_json["words"]
                    current_start = 0
                    current_text = ""
                    
                    # Group words into segments
                    for word in words:
                        if current_text and word["start"] - current_start > 1.5:  # New segment if gap > 1.5s
                            buffer.append({"text": current_text.strip(), "start": current_start})
                            current_text = word["text"]
                            current_start = word["start"]
                        else:
                            current_text += " " + word["text"] if current_text else word["text"]
                    
                    # Add the last segment
                    if current_text:
                        buffer.append({"text": current_text.strip(), "start": current_start})
                        
                else:
                    # Fallback if word timestamps aren't available
                    text = polling_response_json.get("text", "")
                    if text.strip():
                        # Use the filename to estimate start time (chunk_X.wav)
                        try:
                            chunk_num = int(filename.split("_")[1].split(".")[0])
                            start_time = chunk_num * CHUNK_DURATION
                        except (IndexError, ValueError):
                            start_time = 0
                        buffer.append({"text": text, "start": start_time})
                
                break
            elif status == "error":
                print(f"Transcription error: {polling_response_json.get('error')}")
                break
            elif retry_count >= max_retries - 1:
                print("Max polling attempts reached")
                break
            else:
                # Wait a bit before polling again
                retry_count += 1
                time.sleep(2)
                
        print(f"Chunk processing complete, buffer has {len(buffer)} segments")
                
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        import traceback
        traceback.print_exc()

def recorder():
    """Background thread for recording audio"""
    chunk_counter = 0
    print("Recorder thread started")
    
    while True:
        try:
            # Check if we should stop
            if not hasattr(st, "session_state") or not st.session_state.recording:
                print("Recording stopped, exiting recorder thread")
                break
                
            # Collect chunks for CHUNK_DURATION seconds
            chunks = []
            chunk_size = SAMPLE_RATE * CHUNK_DURATION // CHUNK_SIZE
            print(f"Collecting {chunk_size} audio chunks...")
            
            for i in range(chunk_size):
                try:
                    chunk = q.get(timeout=1.0)  # 1 second timeout
                    chunks.append(chunk)
                except queue.Empty:
                    print(f"Queue empty after {i} chunks")
                    if not st.session_state.recording:
                        break
            
            if not chunks:
                print("No audio chunks collected, retrying")
                continue
                
            # Combine chunks and save
            print(f"Processing {len(chunks)} audio chunks")
            audio_data = np.concatenate(chunks)
            filename = f"chunk_{chunk_counter}.wav"
            save_audio_chunk(audio_data, filename)
            
            # Process the chunk
            print(f"Processing audio file: {filename}")
            process_audio_chunk(filename)
            
            # Clean up
            try:
                os.remove(filename)
            except Exception as e:
                print(f"Error removing file {filename}: {e}")
                
            chunk_counter += 1
            
        except Exception as e:
            print(f"Error in recorder thread: {e}")
            import traceback
            traceback.print_exc()
            
            # If we encounter an error, sleep to avoid hammering CPU with errors
            time.sleep(1)
            
            # If recording has stopped, exit the thread
            if not hasattr(st, "session_state") or not st.session_state.recording:
                break

def main():
    # Get AssemblyAI API key
    api_key = os.getenv("ASSEMBLY_API_KEY")
    if not api_key:
        print("Please set ASSEMBLY_API_KEY environment variable")
        return
    
    # Start audio stream
    stream = sd.InputStream(
        callback=callback,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE
    )
    
    # Start recording thread
    record_thread = threading.Thread(target=recorder, daemon=True)
    record_thread.start()
    
    print("Recording started. Press Ctrl+C to stop.")
    
    try:
        with stream:
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        stream.stop()
        stream.close()

if __name__ == "__main__":
    main()
