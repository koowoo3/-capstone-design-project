import sys
sys.path.append('C:\\Users\\kooo\\Documents\\smartfactory\\smartfactory\\sort')

import os
import json
import pathlib
from sort import Sort
import cv2
import torch
import numpy as np
import pandas as pd
import time

output_width = 640
output_height = 480

tracker = Sort()  # Tracker 인스턴스 생성
# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Path to your video file
#video_path = 'C:\\Users\\PC\\Documents\\VIRAT_S_010204_05_000856_000890.mp4'
#video_path = 'C:\\Users\\PC\\Documents\\7-4_cam02_assault01_place04_day_summer.mp4'
video_path = 'C:\\Users\\kooo\\Documents\\video.avi'
# Initialize video capture with the video file
cap = cv2.VideoCapture(video_path)

#rtsp_url = 'rtsp://210.99.70.120:1935/live/cctv050.stream'
#cap = cv2.VideoCapture(rtsp_url)
#cap = cv2.VideoCapture(0)

# Function to save predictions to CSV
def save_preds_to_csv(preds, filename):
    data = []
    for timestep, nodes in preds.items():
        for node, pos in nodes.items():
            data.append([timestep*10, node, pos[0], pos[1]])  # Assuming pos is [2] for [x, y]
    df = pd.DataFrame(data, columns=['Timestep', 'PersonID', 'X', 'Y'])
    df.to_csv(filename, index=False)

# Dictionary to store positions for each person
positions = {}

# Dictionary to store predictions for CSV
preds_dict = {}

try:
    frame_count = 0
    timestep = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (output_width, output_height))
        
        # Convert the captured frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model([rgb_frame], size=640)  # Process the image for detection
        
        output_frame = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
        # Convert results to DataFrame for easier manipulation
        df = results.pandas().xyxy[0]  # Results as DataFrame
        person_df = df[df['class'] == 0]
        
        detections = person_df[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].to_numpy()
        tracks = tracker.update(detections)
        
        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # 노드 데이터 업데이트
            person_id = track_id
            if person_id not in positions:
                positions[person_id] = []
            positions[person_id].append([x_center, y_center])

            # 특별 표시: person_1
            if person_id == "person_1":
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # 녹색 테두리
            
            # Add positions to preds_dict
            if timestep not in preds_dict:
                preds_dict[timestep] = {}
            preds_dict[timestep][person_id] = [x_center, y_center]

            print(preds_dict[timestep])
        timestep += 1
        cv2.imshow('YOLOv5 Object Detection', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream stopped")
finally:
    cap.release()
    cv2.destroyAllWindows()

save_preds_to_csv(preds_dict, 'train.csv')
