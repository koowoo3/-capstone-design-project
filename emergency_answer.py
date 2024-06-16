import requests


def answer(describe_sentence):
    api_key = "api_key"
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
            "text": '''A person is in an emergency situation right now. 
            I will describe the situation, and please create a sentence for reporting it to 911. 
            The output should only be the reporting sentence. and please output the korean sensentence
            describe sentence : 
            ''' + describe_sentence
            },
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    point = response.json()["choices"][0]["message"]["content"]
    #print(point)
    return point

   