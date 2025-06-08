import os
import requests
from dotenv import load_dotenv

load_dotenv()

def send_pushover_alert(title, message):
    payload = {
        "token": os.getenv("PUSHOVER_TOKEN"),
        "user": os.getenv("PUSHOVER_USER"),
        "title": title,
        "message": message
    }
    response = requests.post("https://api.pushover.net/1/messages.json", data=payload)
    if response.status_code != 200:
        print("Pushover failed:", response.text)