import pandas as pd
import numpy as np

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

def derivative_of(x, dt=1, radian=False):
    if radian:
        x = make_continuous_copy(x)

    not_nan_mask = ~np.isnan(x)
    masked_x = x[not_nan_mask]

    if masked_x.shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt

    return dx

# 속도 계산
df['VX'] = df.groupby('PersonID')['X'].transform(lambda x: derivative_of(x.values, dt=1))
df['VY'] = df.groupby('PersonID')['Y'].transform(lambda x: derivative_of(x.values, dt=1))

# 가속도 계산
df['AX'] = df.groupby('PersonID')['VX'].transform(lambda x: derivative_of(x.values, dt=1))
df['AY'] = df.groupby('PersonID')['VY'].transform(lambda x: derivative_of(x.values, dt=1))

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

# 표준화 사전 생성
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
