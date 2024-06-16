#from openai import OpenAI
import base64
import requests
import cv2




# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def gpt_answer(image_path_list):
    api_key = "api_key"
    
    #image_path = "/home/wonyeong/Trajectron-plus-plus/image1.png"

    # Getting the base64 string
    
    base64_image1 = encode_image(image_path_list[0])
    base64_image2 = encode_image(image_path_list[1])
    base64_image3 = encode_image(image_path_list[2])
    base64_image4 = encode_image(image_path_list[3])

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
            "text": '''
            I have provided four images as input,
            each with a frame size of 640x480. 
            I need to predict where the person in each image is likely to move next. 
            Please express the predicted coordinates, considering the frame size.
             '''
            
            },
            {
            "type": "image_url",
            "image_url": {
                "urls": [
                    {"url": f"data:image/jpeg;base64,{base64_image1}"},
                    {"url": f"data:image/jpeg;base64,{base64_image2}"},
                    {"url": f"data:image/jpeg;base64,{base64_image3}"},
                    {"url": f"data:image/jpeg;base64,{base64_image4}"},
                    # Add more image URLs as needed
                ]
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    point = response.json()["choices"][0]["message"]["content"]
    print(point)
    return point
