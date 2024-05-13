import os
import json
import pathlib
from argument_parser import args
from model_registrar import ModelRegistrar
from online_trajectron import OnlineTrajectron
from environment import Environment, Scene
from environment.node import Node 
import visualization as vis
import cv2
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Path to your video file
video_path = 'C:\\Users\\kooo\\Downloads\\New_Sample\\ja-ma\\9_12\\video\\2021-08-01_09-12-00_sun_sunny_out_ja-ma_C0041.mp4'

# Initialize video capture with the video file
cap = cv2.VideoCapture(video_path)

def compute_velocity_and_acceleration(positions):
    # Calculate velocities
    velocities = np.diff(positions, axis=0)
    # Calculate accelerations
    accelerations = np.diff(velocities, axis=0)
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
print(model_dir)
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

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
node_frequency_multiplier = 1 # 이거 뭐 해야할지 모르겠음

attention_radius = dict()
attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
# attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
# attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
# attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

env.attention_radius = attention_radius
# env.robot_type = env.NodeType.VEHICLE



model_registrar = ModelRegistrar(model_dir, args.eval_device)
model_registrar.load_models(iter_num=12)

trajectron = OnlineTrajectron(model_registrar,
                                hyperparams,
                                args.eval_device)

init_timestep = 1
trajectron.set_environment(env, init_timestep)
time_check=0

preds_dict = {}
#scenes = []
#scene = process_scene(ns_scene, env, nusc, data_path)
#scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_id), aug_func=augment)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Initialize positions dictionary for each new frame
        

        # Convert the captured frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model([rgb_frame], size=640)  # Process the image for detection
        
        # Convert results to DataFrame for easier manipulation
        df = results.pandas().xyxy[0]  # Results as DataFrame

        # Filter detections for class 'person' (class ID for 'person' is typically 0 for COCO dataset)
        person_df = df[df['class'] == 0]  # Assuming 'person' class ID is 0

        input_dict = dict()
        for index, row in person_df.iterrows():
            person_id = f"person_{index}"  # Create a unique ID for each person
            x_center = (row['xmin'] + row['xmax']) / 2
            y_center = (row['ymin'] + row['ymax']) / 2

            if person_id not in positions:
                positions[person_id] = []

            positions[person_id].append([x_center, y_center])
        
            if len(positions[person_id]) > 2:
                velocities, accelerations = compute_velocity_and_acceleration(np.array(positions[person_id]))
                if len(velocities) > 0 and len(accelerations) > 0:
                    latest_velocity = velocities[-1]
                    latest_acceleration = accelerations[-1]

                    #print("latest_velocity[0]: ",latest_velocity[0])
                    data_dict = {('position', 'x'): x_center,
                         ('position', 'y'): y_center,
                         ('velocity', 'x'): latest_velocity[0],
                         ('velocity', 'y'): latest_velocity[1],
                         ('acceleration', 'x'): latest_acceleration[0],
                         ('acceleration', 'y'): latest_acceleration[1]}
                    
                    
                    node_data = pd.DataFrame(data_dict, index=[0], columns=data_columns_pedestrian)
                    node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=person_id, data=node_data, frequency_multiplier=node_frequency_multiplier)
                    input_dict[node] = node_data
                    input_dict[node] = node.get(np.array([time_check, time_check]), {"position": ["x", "y"],"velocity": ["x", "y"],"acceleration": ["x", "y"]})



                #node.first_timestep = node_df['frame_id'].iloc[0]
                #scene.nodes.append(node)

        #print(final_output)
        maps = None
        robot_present_and_future = None

        start = time.time()
        dists, preds = trajectron.incremental_forward(input_dict,
                                                        maps,
                                                        prediction_horizon=6,
                                                        num_samples=1,
                                                        robot_present_and_future=robot_present_and_future,
                                                        full_dist=True)
        end = time.time()
        
        preds_dict[time_check] = preds
        time_check = time_check+1
        
        # Render results back on the frame (modifies the images in-place)
        results.render()

        # Convert RGB image to BGR for OpenCV display
        output_frame = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow('YOLOv5 Object Detection', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream stopped")
finally:
    cap.release()
    cv2.destroyAllWindows()


save_preds_to_csv(preds_dict, 'prediction.csv')





