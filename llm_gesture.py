#from openai import OpenAI
import base64
import requests
import cv2
import os



# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def gpt_answer(image_path):
    api_key = "api_key"
    
    #image_path = "/home/wonyeong/Trajectron-plus-plus/image1.png"

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            #"text": 'where is most dangerous place in the image?'
            "text": '''Describe what the person in the image is doing. 
            If the person appears to be collapsed or looks dead, it is an emergency situation. 
            Let me know if you think the person in the image is in an emergency situation. 
            if people is lying down at the bottom, it is also emergency situation.
            If it doesn't look like an emergency, say it's not an emergency.
            Print 1 at the beginning if it is an emergency, and print 0 if it is not an emergency. 
            '''
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    point = response.json()["choices"][0]["message"]["content"]
    #print(point)
    return point

   