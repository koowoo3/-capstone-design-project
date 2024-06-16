import sys
import os
import numpy as np
import pandas as pd
import dill

sys.path.append("../environment")
# 현재 파일의 위치

from environment import Environment, Scene, Node
from utils import maybe_makedirs
from environment import derivative_of

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.4

# train.csv 파일에서 데이터 불러오기
train_csv_path = 'train.csv'
df = pd.read_csv(train_csv_path)

# 데이터 확인
print(df.head())

# 위치 데이터 추출
positions = df[['X', 'Y']]

# 위치 데이터의 평균과 표준편차 계산
position_mean = positions.mean()
position_std = positions.std()

print("Position Mean:\n", position_mean)
print("Position Std:\n", position_std)

# 속도 계산 (차분을 사용하여 연속된 위치 값의 차이 계산)
df['VX'] = df.groupby('PersonID')['X'].diff() / df.groupby('PersonID')['Timestep'].diff()
df['VY'] = df.groupby('PersonID')['Y'].diff() / df.groupby('PersonID')['Timestep'].diff()

# 가속도 계산 (차분을 사용하여 연속된 속도 값의 차이 계산)
df['AX'] = df.groupby('PersonID')['VX'].diff() / df.groupby('PersonID')['Timestep'].diff()
df['AY'] = df.groupby('PersonID')['VY'].diff() / df.groupby('PersonID')['Timestep'].diff()

# 속도와 가속도 데이터에서 NaN 값 제거 (첫 번째 차분으로 인한 NaN 제거)
velocity = df[['VX', 'VY']].dropna()
acceleration = df[['AX', 'AY']].dropna()

# 속도 데이터의 평균과 표준편차 계산
velocity_mean = velocity.mean()
velocity_std = velocity.std()

print("Velocity Mean:\n", velocity_mean)
print("Velocity Std:\n", velocity_std)

# 가속도 데이터의 평균과 표준편차 계산
acceleration_mean = acceleration.mean()
acceleration_std = acceleration.std()

print("Acceleration Mean:\n", acceleration_mean)
print("Acceleration Std:\n", acceleration_std)

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': position_mean['X'], 'std': position_std['X']},
            'y': {'mean': position_mean['Y'], 'std': position_std['Y']}
        },
        'velocity': {
            'x': {'mean': velocity_mean['VX'], 'std': velocity_std['VX']},
            'y': {'mean': velocity_mean['VY'], 'std': velocity_std['VY']}
        },
        'acceleration': {
            'x': {'mean': acceleration_mean['AX'], 'std': acceleration_std['AX']},
            'y': {'mean': acceleration_mean['AY'], 'std': acceleration_std['AY']}
        }
    }
}

print("Standardization Dictionary:\n", standardization)





# standardization = {
#     'PEDESTRIAN': {
#         'position': {
#             'x': {'mean': 0, 'std': 1},
#             'y': {'mean': 0, 'std': 1}
#         },
#         'velocity': {
#             'x': {'mean': 0, 'std': 2},
#             'y': {'mean': 0, 'std': 2}
#         },
#         'acceleration': {
#             'x': {'mean': 0, 'std': 1},
#             'y': {'mean': 0, 'std': 1}
#         }
#     }
# }


def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay}

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


nl = 0
l = 0
maybe_makedirs('../processed')
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
for desired_source in ['data']:
    for data_class in ['train', 'val', 'test']:
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        env.attention_radius = attention_radius

        scenes = []
        data_dict_path = os.path.join('../processed', '_'.join([desired_source, data_class]) + '.pkl')

        for subdir, dirs, files in os.walk(os.path.join('raw', desired_source, data_class)):
            for file in files:
                if file.endswith('.txt'):
                    input_data_dict = dict()
                    full_data_path = os.path.join(subdir, file)
                    print('At', full_data_path)

                    data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

                    data['frame_id'] = data['frame_id'] // 10

                    data['frame_id'] -= data['frame_id'].min()

                    data['node_type'] = 'PEDESTRIAN'
                    data['node_id'] = data['track_id'].astype(str)
                    data.sort_values('frame_id', inplace=True)

                    # Mean Position
                    data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                    data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

                    max_timesteps = data['frame_id'].max()

                    scene = Scene(timesteps=max_timesteps+1, dt=dt, name=desired_source + "_" + data_class, aug_func=augment if data_class == 'train' else None)

                    for node_id in pd.unique(data['node_id']):

                        node_df = data[data['node_id'] == node_id]
                        assert np.all(np.diff(node_df['frame_id']) == 1)

                        node_values = node_df[['pos_x', 'pos_y']].values

                        if node_values.shape[0] < 2:
                            continue

                        new_first_idx = node_df['frame_id'].iloc[0]

                        x = node_values[:, 0]
                        y = node_values[:, 1]
                        vx = derivative_of(x, scene.dt)
                        vy = derivative_of(y, scene.dt)
                        ax = derivative_of(vx, scene.dt)
                        ay = derivative_of(vy, scene.dt)

                        data_dict = {('position', 'x'): x,
                                     ('position', 'y'): y,
                                     ('velocity', 'x'): vx,
                                     ('velocity', 'y'): vy,
                                     ('acceleration', 'x'): ax,
                                     ('acceleration', 'y'): ay}

                        node_data = pd.DataFrame(data_dict, columns=data_columns)
                        node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                        node.first_timestep = new_first_idx

                        scene.nodes.append(node)
                    if data_class == 'train':
                        scene.augmented = list()
                        angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                        for angle in angles:
                            scene.augmented.append(augment_scene(scene, angle))

                    print(scene)
                    scenes.append(scene)
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

print(f"Linear: {l}")
print(f"Non-Linear: {nl}")