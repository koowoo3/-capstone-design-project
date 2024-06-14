import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


import json
import pathlib
from argument_parser import args
from model.model_registrar import ModelRegistrar
from online_trajectron import OnlineTrajectron
from environment.environment import Environment
from environment.node import Node 
from sort.sort import Sort
import cv2
import torch
import numpy as np
import pandas as pd
import time

#torch.nn.Module.dump_patches = True
output_width = 640
output_height = 480

#Koo env
standardization = {
'PEDESTRIAN': {
    'position': {
        'x': {'mean': 0, 'std': 1},
        'y': {'mean': 0, 'std': 1}
    },
    'velocity': {
        'x': {'mean': 0, 'std': 2},
        'y': {'mean': 0, 'std': 2}
    },
    'acceleration': {
        'x': {'mean': 0, 'std': 1},
        'y': {'mean': 0, 'std': 1}
    }
}
}

tracker = Sort()  # Tracker 인스턴스 생성
# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Path to your video file
#video_path = 'C:\\Users\\kooo\\Documents\\video.avi'
#video_path = 'C:\\Users\\PC\\Documents\\7-4_cam02_assault01_place04_day_summer.mp4'
#video_path = 'C:\\Users\\kooo\\Documents\\VIRAT_S_010204_05_000856_000890.mp4'


# Initialize video capture with the video file
#cap = cv2.VideoCapture(video_path)

#rtsp_url = 'rtsp://210.99.70.120:1935/live/cctv050.stream'
#cap = cv2.VideoCapture(rtsp_url)
cap = cv2.VideoCapture(0)

def derivative_of(x, dt=1, radian=False):

    not_nan_mask = ~np.isnan(x)
    masked_x = x[not_nan_mask]

    if masked_x.shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt

    return dx




def compute_velocity_and_acceleration(positions):
    velocities = derivative_of(positions[:, 0], dt=1), derivative_of(positions[:, 1], dt=1)
    accelerations = derivative_of(velocities[0], dt=1), derivative_of(velocities[1], dt=1)
    return velocities, accelerations

# Dictionary to store positions and outputs for each person
final_output = {}
positions = {}

# Function to save predictions to CSV
def save_preds_to_csv(preds, filename):
    data = []
    for timestep, nodes in preds.items():
        for node, positions in nodes.items():
            for pos in positions:
                data.append([node, timestep, pos[0, 0], pos[0, 1]])  # Assuming pos is [1, 2] for [x, y]
    df = pd.DataFrame(data, columns=['Node', 'Timestep', 'X', 'Y'])
    df.to_csv(filename, index=False)


# Choose one of the model directory names under the experiment/*/models folders.
# Possibilities are 'vel_ee', 'int_ee', 'int_ee_me', or 'robot'
model_dir = os.path.join(args.log_dir, 'koo')

# Load hyperparameters from json
config_file = os.path.join(model_dir, args.conf)
if not os.path.exists(config_file):
    raise ValueError('Config json not found!')
with open(config_file, 'r') as conf_json:
    hyperparams = json.load(conf_json)

# Add hyperparams from arguments
hyperparams['dynamic_edges'] = args.dynamic_edges
hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
hyperparams['edge_addition_filter'] = args.edge_addition_filter
hyperparams['edge_removal_filter'] = args.edge_removal_filter
hyperparams['batch_size'] = args.batch_size
hyperparams['k_eval'] = args.k_eval
hyperparams['offline_scene_graph'] = args.offline_scene_graph
hyperparams['incl_robot_node'] = args.incl_robot_node
hyperparams['edge_encoding'] = not args.no_edge_encoding
hyperparams['use_map_encoding'] = args.map_encoding

output_save_dir = os.path.join(model_dir, 'pred_figs')
pathlib.Path(output_save_dir).mkdir(parents=True, exist_ok=True)



data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
node_frequency_multiplier = 1 

attention_radius = dict()
attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0

env.attention_radius = attention_radius



model_registrar = ModelRegistrar(model_dir, args.eval_device)
model_registrar.load_models(iter_num=100)  #12였음

trajectron = OnlineTrajectron(model_registrar,
                                hyperparams,
                                args.eval_device)

init_timestep = 1
trajectron.set_environment(env, init_timestep)
time_check=0

preds_dict = {}

try:
    frame_count = 0
    timestep=0
    cnt = 0
    while True:
        cnt += 1
        ret, frame = cap.read()
        if cnt % 5 == 0:
            
            if not ret:
                break
            frame_count += 1

            # Initialize positions dictionary for each new frame
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
            
            input_dict = dict()
            for track in tracks:
                
                x1, y1, x2, y2, track_id = track.astype(int)
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                # 노드 데이터 업데이트
                person_id = f"person_{track_id}"
                if person_id not in positions:
                    positions[person_id] = []
                positions[person_id].append([x_center, y_center])

                if len(positions[person_id]) > 2:
                    velocities, accelerations = compute_velocity_and_acceleration(np.array(positions[person_id]))
                    if len(velocities) > 0 and len(accelerations) > 0:
                        latest_velocity = velocities[-1]
                        latest_acceleration = accelerations[-1]
                        data_dict = {
                            ('position', 'x'): x_center,
                            ('position', 'y'): y_center,
                            ('velocity', 'x'): latest_velocity[0],
                            ('velocity', 'y'): latest_velocity[1],
                            ('acceleration', 'x'): latest_acceleration[0],
                            ('acceleration', 'y'): latest_acceleration[1]
                        }

                    
                    node_data = pd.DataFrame(data_dict, index=[0], columns=data_columns_pedestrian)
                    node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=person_id, data=node_data, frequency_multiplier=node_frequency_multiplier)
                    input_dict[node] = node_data
                    input_dict[node] = node.get(np.array([time_check, time_check]), {"position": ["x", "y"],"velocity": ["x", "y"],"acceleration": ["x", "y"]})
                    
                    # 노드 정보 출력 (옵션)
                    print(data_dict)

                # # 특별 표시: person_6
                # if person_id == "person_1":
                #     cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # 녹색 테두리
                #     #print(data_dict)
                    
            maps = None
            robot_present_and_future = None

            start = time.time()
            dists, preds = trajectron.incremental_forward(input_dict,
                                                            maps,
                                                            prediction_horizon=30,
                                                            num_samples=6,
                                                            robot_present_and_future=robot_present_and_future,
                                                            full_dist=True)
            end = time.time()
            
            print(preds)
            
            for person_id, predictions in preds.items():
                predicted_positions = predictions[0]  # 첫 번째 샘플의 예측 경로를 가져옴
                for step in range(predicted_positions.shape[1]):
                    x, y = predicted_positions[0, step]  # 예측된 각 시간 단계의 x, y 좌표
                    cv2.circle(output_frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # 녹색 원으로 표시
                
        
            #time_check = time_check+1
                cv2.imshow('YOLOv5 Object Detection', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("Stream stopped")
finally:
    cap.release()
    cv2.destroyAllWindows()


#save_preds_to_csv(preds_dict, 'prediction.csv')





