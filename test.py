# import camera
# import time
# import llm_init
# import llm_gesture
# import cv2
# import llm_traject_prediction
# import threading
# import thread_utils


# cnt = 0
# motion_check = 0
# for i in camera.get_smartphone_camera():
#     cnt += 1 
#     # cv2.rectangle(i, (120, 120), (240, 240), (0, 255, 0), 2)
#     # Draw a point on the image
#     #cv2.circle(i, (320, 240), 5, (255, 0, 0), -1)
#     #cv2.imshow("Smartphone Camera", i)
#     cv2.imwrite("frame.png", i)
#     if cnt == 1:    
#         x,y,w,h = llm_init.gpt_answer('frame.png')
#         print(x,y,w,h)
#     cv2.rectangle(i, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     cv2.imshow("Smartphone Camera", i)
#     image = cv2.imread("frame.png")
#     height, width, _ = image.shape
    
#     # gesture 처리 하는 부분
#     if cnt % 50 == 0:
#         gesture_thread = threading.Thread(target=thread_utils.process_gesture)
#         gesture_thread.start()
    
    
    
    
    
    
    
    
    
    
#     #객체의 위치 예측 파트
#     '''
#     motion_check += 1
#     cv2.imwrite("gesture" + str(motion_check)+"png", i)
#     if motion_check == 4:
#         image_list = []
#         image_list.append('gesture1.png')
#         image_list.append('gesture2.png')
#         image_list.append('gesture3.png')
#         image_list.append('gesture4.png')
#         llm_traject_prediction.gpt_answer(image_list)
#     ''' 
        
    
    
#     #image size : 640 x 480


import cv2
from predict_class import TrajectoryPredictor
import os

video_path = 'C:\\Users\\kooo\\Documents\\VIRAT_S_010204_05_000856_000890.mp4'
cap = cv2.VideoCapture(video_path)

model_dir = os.path.join(os.getcwd(), 'koo')
config_file = os.path.join(model_dir, 'config.json')
trajectory_predictor = TrajectoryPredictor(video_path, model_dir, config_file)

while True:
    ret, frame = cap.read()
    output_frame, predictions = trajectory_predictor.process_frame(frame)
    cv2.imshow('Trajectory Prediction', output_frame)
    print(predictions)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()