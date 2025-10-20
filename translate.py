import os
import requests

with open('api_key.txt', 'r') as f:
    os.environ["gnlp"] = f.read().strip()

gnlp = os.environ["gnlp"]


def translate(text, target_language):
    url = "https://translation-api.ghananlp.org/v1/translate"

    headers = {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Ocp-Apim-Subscription-Key': gnlp,    # put your API key here
    }

    data = {
        'in': text,
        'lang': target_language
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
         return response.json()
    else:
         return f"An error occurred: Error {response.status_code} - {response.text}"  
     
