import sounddevice as sd
import wavio
import whisper
import torch
import json
import random
import requests
import os

# Configuration
AUDIO_FILENAME = "input.wav"
OUTPUT_FOLDER = "images"

# Load Whisper model once globally
WHISPER_MODEL = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

# Example array of items
ITEMS = ["T-Shirt", "Cargo Pants", "Quilt Jacket", "Men's Sneakers",
         "Hawaiian Shirt", "Wide-brim Hat", "Floral Skirt"]

# Record audio from the microphone
def record_audio(filename=AUDIO_FILENAME, duration=5, samplerate=44100):
    print("Recording... Speak now!")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    wavio.write(filename, recording, samplerate, sampwidth=2)
    print(f"Audio saved as {filename}")

# Transcribe audio using Whisper
def transcribe_audio(filepath=AUDIO_FILENAME):
    audio = whisper.load_audio(filepath)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(WHISPER_MODEL.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(WHISPER_MODEL, mel, options)
    return result.text.lower()

# Select ONE item
def select_item(transcription, items=ITEMS):
    matches = [item for item in items if item.lower() in transcription]
    if matches:
        return random.choice(matches)
    else:
        return random.choice(items)

# Fetch 8 images for the selected item (using Unsplash free API as example)
def fetch_images(query, num_images=8):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    urls = []
    for i in range(num_images):
        # This uses Unsplash's source API (no key required, random images)
        url = f"https://source.unsplash.com/600x600/?{query.replace(' ', '%20')}&sig={i}"
        urls.append(url)
        # If you want to download the files locally:
        response = requests.get(url)
        if response.status_code == 200:
            filename = os.path.join(OUTPUT_FOLDER, f"{query}_{i+1}.jpg")
            with open(filename, "wb") as f:
                f.write(response.content)
    return urls

def main(record_duration=5):
    # Step 1: Record audio
    record_audio(duration=record_duration)
    
    # Step 2: Transcribe audio
    transcription = transcribe_audio()
    print("Transcription:", transcription)
    
    # Step 3: Select one item
    selected_item = select_item(transcription)
    print("Selected item:", selected_item)
    
    # Step 4: Fetch 8 images of that item
    image_urls = fetch_images(selected_item, num_images=8)
    print("Fetched 8 images")
    
    # Step 5: Save transcription + item + images
    data = {
        "transcription": transcription,
        "selected_item": selected_item,
        "images": image_urls
    }
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("Saved output.json")

if __name__ == "__main__":
    main(record_duration=5)
