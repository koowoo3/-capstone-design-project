# import pandas as pd
# import matplotlib.pyplot as plt

# # CSV 파일 로드
# df = pd.read_csv('path_data.csv')

# # 시각화
# fig, ax = plt.subplots(figsize=(10, 10))
# for person_id, group in df.groupby('Person ID'):
#     ax.plot(group['X'], group['Y'], marker='o', linestyle='-', label=person_id)

# ax.set_xlabel('X position')
# ax.set_ylabel('Y position')
# ax.set_title('Movement Paths of Detected Persons')
# ax.legend()
# plt.show()





#전체 출력하고 싶을때
import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions_from_csv(filename):
    # CSV 파일 로드
    df = pd.read_csv(filename)

    # 시각화 설정
    plt.figure(figsize=(10, 10))
    groups = df.groupby('Node')
    for name, group in groups:
        plt.plot(group['X'], group['Y'], marker='o', linestyle='-', label=name)

    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Predicted Paths for Each Node')
    plt.legend()
    plt.grid(True)
    plt.show()

# 예제 사용
plot_predictions_from_csv('prediction.csv')

# import matplotlib.pyplot as plt
# import pandas as pd
# def plot_specific_node_predictions_from_csv(filename, specific_node):
#     df = pd.read_csv(filename)
#     df = df[df['Node'] == specific_node]
    
#     plt.figure(figsize=(10, 10))
#     plt.plot(df['X'], df['Y'], marker='o', linestyle='-', label=specific_node)
    
#     plt.xlabel('X position')
#     plt.ylabel('Y position')
#     plt.title(f'Predicted Path for {specific_node}')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # 예제 사용
# specific_node = 'PEDESTRIAN/a67af729352f44cf8872524897cdbeab'
# plot_specific_node_predictions_from_csv('predictions.csv', specific_node)
