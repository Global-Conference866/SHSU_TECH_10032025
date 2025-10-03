import socketserver
import sounddevice as sd
import wavio
import whisper
import torch
import json
import random
import requests
from google import genai
from http.server import BaseHTTPRequestHandler

# --- CONFIG ---
GEMINI_KEY = "[REDACTED]" 
 
# Example array of items
ITEMS = ["T-Shirt", "Cargo Pants", "Quilt Jacket", "Men's Sneakers", "Cargo Shorts"
         "Hawaiian Shirt", "Wide-brim Hat", "Floral Skirt"]

# --- Ask Gemini ---
def ask_gemini(prompt):
    client = genai.Client(api_key=GEMINI_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents= prompt + " . Given the array " + ''.join(ITEMS) + " select the item that most closely matches the previous transcript. Your response should be a singular integer, the index of the item in the array. Assume that we begin indexing from 1."
    )
    return response.text.strip()

# Select ONE item
def select_item(transcription, items=ITEMS):
    matches = [item for item in items if item.lower() in transcription]
    if matches:
        return random.choice(matches)
    else:
        return random.choice(items)

def GetClothingID(transcription):
    selected_item = ask_gemini(transcription)
    print("Selected item:", selected_item)
    return selected_item
 
class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
 
        message = "GET Request Response"
        self.wfile.write(bytes(message, "utf8"))
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
 
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        transcription = post_data.decode('utf-8')
        print(transcription)
        selected_item = GetClothingID(transcription)
 
        message = selected_item
        self.wfile.write(bytes(message, "utf8"))


httpd = socketserver.TCPServer(("", 8080), MyHandler)
httpd.serve_forever()
 