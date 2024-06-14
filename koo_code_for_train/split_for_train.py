# import sys
# import os
# import pandas as pd
# import numpy as np

# # 현재 파일의 위치
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # environment 모듈의 절대 경로
# environment_path = os.path.abspath(os.path.join(current_dir, '..', 'environment'))

# # sys.path에 environment 경로 추가
# sys.path.append(environment_path)

# # CSV 파일 읽기
# data = pd.read_csv('train.csv')

# # 전체 프레임의 범위 계산
# min_timestep = data['Timestep'].min()
# max_timestep = data['Timestep'].max()
# total_timesteps = max_timestep - min_timestep + 1

# # 비율에 따라 데이터 나누기
# train_ratio = 0.7
# val_ratio = 0.15
# test_ratio = 0.15

# train_end = min_timestep + int(total_timesteps * train_ratio) - 1
# val_end = train_end + int(total_timesteps * val_ratio)

# # Debugging AssertionError and filling missing frames with 10-step intervals
# filled_data = pd.DataFrame()
# for person_id in data['PersonID'].unique():
#     node_df = data[data['PersonID'] == person_id]
#     min_time = node_df['Timestep'].min()
#     max_time = node_df['Timestep'].max()
    
#     # Create a range of timesteps with a step of 10
#     full_range = pd.DataFrame({'Timestep': np.arange(min_time, max_time + 1, 10)})
    
#     # Merge with the original data and forward fill missing values
#     node_df = full_range.merge(node_df, on='Timestep', how='left')
#     node_df.ffill(inplace=True)
#     node_df['PersonID'] = person_id  # Ensure the PersonID is correctly assigned after the merge
#     filled_data = pd.concat([filled_data, node_df])

# # 전체 데이터를 정렬 (Timestep, PersonID 순)
# filled_data.sort_values(by=['Timestep', 'PersonID'], inplace=True)

# # 데이터를 분할
# train_data_filled = filled_data[(filled_data['Timestep'] >= min_timestep) & (filled_data['Timestep'] <= train_end)]
# val_data_filled = filled_data[(filled_data['Timestep'] > train_end) & (filled_data['Timestep'] <= val_end)]
# test_data_filled = filled_data[(filled_data['Timestep'] > val_end) & (filled_data['Timestep'] <= max_timestep)]

# # 데이터 저장할 폴더 생성
# output_dir = 'data'
# train_dir = os.path.join(output_dir, 'train')
# val_dir = os.path.join(output_dir, 'val')
# test_dir = os.path.join(output_dir, 'test')

# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)

# # 각 데이터셋을 TXT 파일로 저장 (탭으로 구분), 열 헤더를 제거
# train_data_filled.to_csv(os.path.join(train_dir, 'train_data.txt'), sep='\t', index=False, header=False)
# val_data_filled.to_csv(os.path.join(val_dir, 'val_data.txt'), sep='\t', index=False, header=False)
# test_data_filled.to_csv(os.path.join(test_dir, 'test_data.txt'), sep='\t', index=False, header=False)

# print(f'Filled train data saved in {train_dir} folder.')
# print(f'Filled validation data saved in {val_dir} folder.')
# print(f'Filled test data saved in {test_dir} folder.')


import sys
import os
import pandas as pd
import numpy as np

# 현재 파일의 위치
current_dir = os.path.dirname(os.path.abspath(__file__))

# environment 모듈의 절대 경로
environment_path = os.path.abspath(os.path.join(current_dir, '..', 'environment'))

# sys.path에 environment 경로 추가
sys.path.append(environment_path)

# TXT 파일 읽기
data = pd.read_table('output.txt', sep='\t', header=None, names=['Timestep', 'PersonID', 'X', 'Y'])

# 전체 프레임의 범위 계산
min_timestep = data['Timestep'].min()
max_timestep = data['Timestep'].max()
total_timesteps = max_timestep - min_timestep + 1

# 비율에 따라 데이터 나누기
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

train_end = min_timestep + int(total_timesteps * train_ratio) - 1
val_end = train_end + int(total_timesteps * val_ratio)

# Debugging AssertionError and filling missing frames with 10-step intervals
filled_data = pd.DataFrame()
for person_id in data['PersonID'].unique():
    node_df = data[data['PersonID'] == person_id]
    min_time = node_df['Timestep'].min()
    max_time = node_df['Timestep'].max()
    
    # Create a range of timesteps with a step of 10
    full_range = pd.DataFrame({'Timestep': np.arange(min_time, max_time + 1, 10)})
    
    # Merge with the original data and forward fill missing values
    node_df = full_range.merge(node_df, on='Timestep', how='left')
    node_df.ffill(inplace=True)
    node_df['PersonID'] = person_id  # Ensure the PersonID is correctly assigned after the merge
    filled_data = pd.concat([filled_data, node_df])

# 전체 데이터를 정렬 (Timestep, PersonID 순)
filled_data.sort_values(by=['Timestep', 'PersonID'], inplace=True)

# 데이터를 분할
train_data_filled = filled_data[(filled_data['Timestep'] >= min_timestep) & (filled_data['Timestep'] <= train_end)]
val_data_filled = filled_data[(filled_data['Timestep'] > train_end) & (filled_data['Timestep'] <= val_end)]
test_data_filled = filled_data[(filled_data['Timestep'] > val_end) & (filled_data['Timestep'] <= max_timestep)]

# 데이터 저장할 폴더 생성
output_dir = 'data'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 각 데이터셋을 TXT 파일로 저장 (탭으로 구분), 열 헤더를 제거
train_data_filled.to_csv(os.path.join(train_dir, 'train_data.txt'), sep='\t', index=False, header=False)
val_data_filled.to_csv(os.path.join(val_dir, 'val_data.txt'), sep='\t', index=False, header=False)
test_data_filled.to_csv(os.path.join(test_dir, 'test_data.txt'), sep='\t', index=False, header=False)

print(f'Filled train data saved in {train_dir} folder.')
print(f'Filled validation data saved in {val_dir} folder.')
print(f'Filled test data saved in {test_dir} folder.')
