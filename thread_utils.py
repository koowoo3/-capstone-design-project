import llm_gesture
import speech_recognition as sr
from gtts import gTTS
import os
import time
import playsound
import emergency_answer


def process_gesture():
    answer = llm_gesture.gpt_answer('frame.png')
    print(answer)
    print()
    if answer[0] == '0':
        #speak(answer)
        pass
    
    elif answer[0] == '1':
        print("It is an emergency so 신고 프로세스 시작")
        result = emergency_answer.answer(answer)
        speak(result)


def answer_trajection():
    point = '위험해요 조심하세요'
    speak1(point)
    return point


def speak(text):
     tts = gTTS(text=text, lang='ko')
     filename='voice.mp3'
     tts.save(filename)
     playsound.playsound(filename)



def speak1(text):
     tts = gTTS(text=text, lang='ko')
     filename='voice1.mp3'
     tts.save(filename)
     playsound.playsound(filename)