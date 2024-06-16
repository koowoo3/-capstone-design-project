import camera
import time
import llm_init
import llm_gesture
import cv2
import llm_traject_prediction
import threading
import thread_utils
from predict_class import TrajectoryPredictor
import os
import socket_tcp

model_dir = os.path.join(os.getcwd(), 'koo')
config_file = os.path.join(model_dir, 'config.json')
trajectory_predictor = TrajectoryPredictor(model_dir, config_file)


cnt = 0
motion_check = 0
for i in camera.get_smartphone_camera():
    cnt += 1 
    prediction_list = []
    
    
    cv2.imwrite("frame.png", i)
    if cnt % 1 == 0:
        output_frame, predictions = trajectory_predictor.process_frame(i)
        if predictions is not None and len(predictions) != 0:  # predicted_positions가 비어있는지 확인
            for position in predictions:  # predicted_positions의 값을 순회
                # x1, y1 = position
                # prediction_list.append((x1, y1))  # prediction_list에 추가
                # cv2.circle(i, (int(x1), int(y1)), 5, (0, 255, 0), -1)
                for x1, y1 in position:
                    prediction_list.append((x1, y1))  # prediction_list에 추가
                    cv2.circle(i, (int(x1), int(y1)), 5, (0, 255, 0), -1)


                # 예측된 값이 지정된 영역 내에 있는지 확인
            if x <= x1 <= x + w and y <= y1 <= y + h:
                #thread_utils.answer_trajection()
                sound_thread = threading.Thread(target= thread_utils.answer_trajection)
                sound_thread.start()
            else:
                pass
                
    if cnt == 1:
        x,y,w,h = llm_init.gpt_answer('frame.png')
        print(x,y,w,h)
    cv2.rectangle(i, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #print("predictions: " + str(predictions))
    cv2.imshow("Smartphone Camera", i)
    image = cv2.imread("frame.png")
    height, width, _ = image.shape
    
    # gesture 처리 하는 부분
    if cnt % 50 == 0:
        gesture_thread = threading.Thread(target=thread_utils.process_gesture)
        gesture_thread.start()
    
    # 경로 예측 후 신고 부분
    
    
    
    
    
    
    
    
    
    #객체의 위치 예측 파트
    '''
    motion_check += 1
    cv2.imwrite("gesture" + str(motion_check)+"png", i)
    if motion_check == 4:
        image_list = []
        image_list.append('gesture1.png')
        image_list.append('gesture2.png')
        image_list.append('gesture3.png')
        image_list.append('gesture4.png')
        llm_traject_prediction.gpt_answer(image_list)
    ''' 
        
    
    
    #image size : 640 x 480
    
    
    
           
    # cv2.rectangle(i, (120, 120), (240, 240), (0, 255, 0), 2)
    # Draw a point on the image
    #cv2.circle(i, (320, 240), 5, (255, 0, 0), -1)
    #cv2.imshow("Smartphone Camera", i)
    # Convert the image to bytes
    
    
    
    # _, img_bytes = cv2.imencode('.png', i)
    # print(img_bytes)
    # print(type(img_bytes))
    # Send the image over UDP
    # if cnt == 1:
    #     socket_thread = threading.Thread(target=socket_tcp.tcp_client)
    #     socket_thread.start()
    #socket_tcp.tcp_client(img_bytes)
    
    