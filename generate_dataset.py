
import json
import pandas as pd
import numpy as np

import base64
import requests

import re
import os

with open('example_data/index.txt', 'r') as f:
    index = f.read().splitlines()

index_folder = ['images/' + i.split('_')[0] +'/'+ i.split('_')[1] for i in index]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def llm(base64_image, ocr):
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "use 5~8 words to describe the font style and color of following words in image:\n{}\n reply me with json format {{key: description, ..}}".format(ocr)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

def extract(cleaned_string):
    pattern = r'{.*}'
    match = re.search(pattern, cleaned_string, re.DOTALL)  # 使用 re.DOTALL 匹配多行內容
    if match:
        extracted_dict = match.group()
        try:
            json_dict = json.loads(extracted_dict)
            return json_dict
        except json.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            return None
    else:
        return None

def image_description(image_path):
    try:
        base64_image = encode_image(image_path+'/image.jpg')
    except:
        base64_image = encode_image(image_path+'/image.png')

    with open(image_path+'/ocr.txt', 'r') as f:
        ocr = f.read().splitlines()
        ocr = [ocr[i].split(' ')[0] for i in range(len(ocr))]
    ocr = ', '.join(ocr)

    llm_res = llm(base64_image, ocr)
    llm_res = llm_res.replace("```json", "").replace("```", "").strip()
    description = extract(llm_res)

    with open(image_path+'/description.json', 'w') as f:
        json.dump(description, f)

for i in range(len(index_folder)):
    image_description(index_folder[i])




