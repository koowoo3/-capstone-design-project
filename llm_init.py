#from openai import OpenAI
import base64
import requests
import cv2




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
            "text": '''I want to find the dangerous area in the image I sent. 
            I want to find the dangerous area in the form of a bounding box. 
            The entire image has a width of 640 and a height of 480. 
            In this case, find x, y, w, h. x and y are the coordinates of 
            the top left corner of the bounding box. w and h are the width and 
            height. When outputting, just output the values of x, y, w, h, 
            like this: '200,300,120,120'. '''
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
    #return point
    if point[0] not in ('0','1','2','3','4','5','6','7','8','9'):
        return 240,200,140,150
    
    x, y, w, h = map(float, point.split(','))

    x = int(round(x))
    y = int(round(y))
    w = int(round(w))
    h = int(round(h))

    return x, y, w, h
    