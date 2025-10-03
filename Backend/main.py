# pip install sounddevice wavio whisper torch google-generativeai google-api-python-client requests

import sounddevice as sd
import wavio
import whisper
import torch
from google import genai
from googleapiclient.discovery import build
import json
import requests

# --- CONFIG ---
GEMINI_KEY = "[REDACTED]"          # Gemini API key
CUSTOM_SEARCH_ENGINE_ID = "[REDACTED]"         # Custom Search Engine ID
AUDIO_FILENAME = "input.wav"                # Temporary recorded audio file

# --- Record microphone audio ---
def record_audio(filename=AUDIO_FILENAME, duration=5, samplerate=44100):
    print(" Recording... Speak now!")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    wavio.write(filename, recording, samplerate, sampwidth=2)
    print(f"Audio saved as {filename}")

# --- Transcribe with Whisper ---
def transcribe_audio(filepath=AUDIO_FILENAME):
    model = whisper.load_model("turbo", device="cuda" if torch.cuda.is_available() else "cpu")
    audio = whisper.load_audio(filepath)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

# --- Ask Gemini ---
def ask_gemini(prompt):
    client = genai.Client(api_key=GEMINI_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt + " . Answer in 3 sentences or less."
    )
    return response.text.strip()


# --- Google Image Search ---
def google_image_search(query, num_results=8):
    service = build("customsearch", "v1", developerKey=GEMINI_KEY)
    res = service.cse().list(
        q=query,
        cx=CUSTOM_SEARCH_ENGINE_ID,
        searchType="image",
        num=num_results
    ).execute()
    
    if "items" in res:
        return [item["link"] for item in res["items"]]
    return []

# --- Download image ---
def download_images(urls):
    saved_files = []
    for i, url in enumerate(urls, start=1):
        filename = f"image_{i}.jpg"
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                saved_files.append(filename)
                print(f"Saved {filename}")
            else:
                print(f" Failed to download {url} ({response.status_code})")
        except Exception as e:
            print(f" Error downloading {url}: {e}")
    return saved_files


# --- MAIN WORKFLOW ---
if __name__ == "__main__":
    # Step 1: Record voice
    record_audio(duration=5)
    
    # Step 2: Transcribe
    question = transcribe_audio()
    print("Transcription:", question)
    
    # Step 3: Ask Gemini
    answer = ask_gemini(question)
    print("Response:", answer)
    
    # Step 4: Search for images (8 results)
    image_links = google_image_search(question, num_results=8)
    
    # Step 5: Download images
    saved_files = download_images(image_links)
    
    # Step 6: Save to JSON
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump({
            "question": question,
            "answer": answer,
            "image_urls": image_links,
            "saved_files": saved_files
        }, f, indent=4)
    print("Saved output.json")
    
    input("\nPress Enter to exit...")
