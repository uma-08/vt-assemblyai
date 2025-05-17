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

#ASSEMBLY AI key = a7521cb262b842588f05caf0fbee9d76
# Global variables
q = queue.Queue()
buffer: List[Dict] = []
CHUNK_DURATION = 30  # seconds
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
            "authorization": api_key,
            "content-type": "application/json"
        }

        # Step 1: Upload the audio file
        with open(filename, "rb") as f:
            response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers={"authorization": api_key},
                data=f
            )

        if response.status_code != 200:
            print(f"Error uploading audio: {response.text}")
            return

        upload_url = response.json()["upload_url"]

        # Step 2: Submit the transcription request
        transcription_request = {
            "audio_url": upload_url,
            # Optional: boost certain words
            "word_boost": ["meeting", "transcript", "important"],
            "punctuate": True,
            "format_text": True,
            "speaker_labels": False,  # Set to True if you want speaker diarization
            "word_timestamps": True    # Get timestamps for each word
        }

        response = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            json=transcription_request,
            headers=headers
        )

        if response.status_code != 200:
            print(f"Error submitting transcription request: {response.text}")
            return

        transcript_id = response.json()["id"]

        # Step 3: Poll for the transcription result
        polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

        while True:
            polling_response = requests.get(polling_endpoint, headers=headers)
            polling_response_json = polling_response.json()

            if polling_response_json["status"] == "completed":
                # Process the completed transcription
                if "words" in polling_response_json:
                    words = polling_response_json["words"]
                    current_start = 0
                    current_text = ""

                    # Group words into segments
                    for word in words:
                        # New segment if gap > 1.5s
                        if current_text and word["start"] - current_start > 1.5:
                            buffer.append(
                                {"text": current_text.strip(), "start": current_start})
                            current_text = word["text"]
                            current_start = word["start"]
                        else:
                            current_text += " " + word["text"]

                    # Add the last segment
                    if current_text:
                        buffer.append(
                            {"text": current_text.strip(), "start": current_start})

                else:
                    # Fallback if word timestamps aren't available
                    text = polling_response_json["text"]
                    if text.strip():
                        # Use the filename to estimate start time (chunk_X.wav)
                        chunk_num = int(filename.split("_")[1].split(".")[0])
                        start_time = chunk_num * CHUNK_DURATION
                        buffer.append({"text": text, "start": start_time})

                break
            elif polling_response_json["status"] == "error":
                print(f"Transcription error: {polling_response_json['error']}")
                break
            else:
                # Wait a bit before polling again
                time.sleep(2)

        print(f"Transcription completed with {len(buffer)} segments")

    except Exception as e:
        print(f"Error processing audio chunk: {e}")


def recorder():
    """Background thread for recording audio"""
    chunk_counter = 0
    while True:
        try:
            # Collect chunks for CHUNK_DURATION seconds
            chunks = []
            for _ in range(SAMPLE_RATE * CHUNK_DURATION // CHUNK_SIZE):
                chunk = q.get()
                chunks.append(chunk)

            # Combine chunks and save
            audio_data = np.concatenate(chunks)
            filename = f"chunk_{chunk_counter}.wav"
            save_audio_chunk(audio_data, filename)

            # Process the chunk
            process_audio_chunk(filename)

            # Clean up
            os.remove(filename)
            chunk_counter += 1

        except Exception as e:
            print(f"Error in recorder thread: {e}")


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
