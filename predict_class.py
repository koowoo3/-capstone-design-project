import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import json
import pathlib
import cv2
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
#import dill
from argument_parser import args
from model.model_registrar import ModelRegistrar
from online_trajectron import OnlineTrajectron
from environment.environment import Environment
from environment.node import Node 
from sort.sort import Sort
import visualization as vis

class TrajectoryPredictor:
    def __init__(self, model_dir, config_file, output_width=640, output_height=480):
        self.output_width = output_width
        self.output_height = output_height
        self.tracker = Sort()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        with open(config_file, 'r') as conf_json:
            self.hyperparams = json.load(conf_json)
        
        self.hyperparams.update({
            'dynamic_edges': args.dynamic_edges,
            'edge_state_combine_method': args.edge_state_combine_method,
            'edge_influence_combine_method': args.edge_influence_combine_method,
            'edge_addition_filter': args.edge_addition_filter,
            'edge_removal_filter': args.edge_removal_filter,
            'batch_size': args.batch_size,
            'k_eval': args.k_eval,
            'offline_scene_graph': args.offline_scene_graph,
            'incl_robot_node': args.incl_robot_node,
            'edge_encoding': not args.no_edge_encoding,
            'use_map_encoding': args.map_encoding
        })

        self.output_save_dir = os.path.join(model_dir, 'pred_figs')
        pathlib.Path(self.output_save_dir).mkdir(parents=True, exist_ok=True)

        self.env = self._initialize_environment()
        self.model_registrar = ModelRegistrar(model_dir, args.eval_device)
        self.model_registrar.load_models(iter_num=100)
        self.trajectron = OnlineTrajectron(self.model_registrar, self.hyperparams, args.eval_device)
        self.trajectron.set_environment(self.env, init_timestep=1)

        self.positions = {}
        self.preds_dict = {}
        self.data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
        self.time_check = 0

    def _initialize_environment(self):
        standardization = {
            'PEDESTRIAN': {
                'position': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}},
                'velocity': {'x': {'mean': 0, 'std': 2}, 'y': {'mean': 0, 'std': 2}},
                'acceleration': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}}
            }
        }
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        env.attention_radius = attention_radius
        return env

    def derivative_of(self, x, dt=1, radian=False):
        not_nan_mask = ~np.isnan(x)
        masked_x = x[not_nan_mask]
        if masked_x.shape[-1] < 2:
            return np.zeros_like(x)
        dx = np.full_like(x, np.nan)
        dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt
        return dx

    def compute_velocity_and_acceleration(self, positions):
        velocities = self.derivative_of(positions[:, 0], dt=1), self.derivative_of(positions[:, 1], dt=1)
        accelerations = self.derivative_of(velocities[0], dt=1), self.derivative_of(velocities[1], dt=1)
        return velocities, accelerations

    def save_preds_to_csv(self, preds, filename):
        data = []
        for timestep, nodes in preds.items():
            for node, positions in nodes.items():
                for pos in positions:
                    data.append([node, timestep, pos[0, 0], pos[0, 1]])  # Assuming pos is [1, 2] for [x, y]
        df = pd.DataFrame(data, columns=['Node', 'Timestep', 'X', 'Y'])
        df.to_csv(filename, index=False)

    def process_frame(self, frame):

        frame = cv2.resize(frame, (self.output_width, self.output_height))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model([rgb_frame], size=640)
        output_frame = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
        df = results.pandas().xyxy[0]
        person_df = df[df['class'] == 0]
        detections = person_df[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].to_numpy()
        tracks = self.tracker.update(detections)
        input_dict = dict()
        
        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            person_id = f"person_{track_id}"
            if person_id not in self.positions:
                self.positions[person_id] = []
            self.positions[person_id].append([x_center, y_center])
            if len(self.positions[person_id]) > 2:
                velocities, accelerations = self.compute_velocity_and_acceleration(np.array(self.positions[person_id]))
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
                    node_data = pd.DataFrame(data_dict, index=[0], columns=self.data_columns_pedestrian)
                    node = Node(node_type=self.env.NodeType.PEDESTRIAN, node_id=person_id, data=node_data, frequency_multiplier=1)
                    input_dict[node] = node_data
                    input_dict[node] = node.get(np.array([self.time_check, self.time_check]), {"position": ["x", "y"],"velocity": ["x", "y"],"acceleration": ["x", "y"]})
        
        
        maps = None
        robot_present_and_future = None
        k = False
        _, preds = self.trajectron.incremental_forward(input_dict, maps, prediction_horizon=30, num_samples=6, robot_present_and_future=robot_present_and_future, full_dist=True)
        
        # 모든 정보 저장하고 싶을때
        # self.preds_dict.update(preds)
        all_predicted_positions = [] 
        for person_id, predictions in preds.items():
            predicted_positions = predictions[0]
            k= True

            person_positions = []
            # 모든 경로 다 출력하고 싶을 때
            for step in range(predicted_positions.shape[1]):
                x, y = predicted_positions[0, step]
                person_positions.append((int(x), int(y)))
                cv2.circle(output_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            all_predicted_positions.append(person_positions)
            # 최종 경로만 출력하고 싶을 때
            # final_position = predicted_positions[0, -1] 
            # x, y = final_position
            # cv2.circle(output_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # # 예측 결과 저장 x
        #final_positions = {}
        #for person_id, predictions in preds.items():
            #predicted_positions = predictions[0]
            #final_position = predicted_positions[0, -1]  # Get the final position (last time step)
            #final_positions[person_id] = final_position
            #x, y = final_position
            #cv2.circle(output_frame, (int(x), int(y)), 5, (0, 255, 0), -1)


        #return output_frame, self.preds_dict
        #return output_frame, final_positions
        if k == True:
            k = False
            return output_frame, all_predicted_positions
        else:
            return output_frame, None

if __name__ == "__main__":

    model_dir = os.path.join(os.getcwd(), 'koo')
    config_file = os.path.join(model_dir, 'config.json')
    trajectory_predictor = TrajectoryPredictor(model_dir, config_file)
    predictions = trajectory_predictor.process_video()
    print(predictions)
