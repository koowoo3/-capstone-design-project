# import cv2
# import torch
# import numpy as np
# import time

# # Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Path to your video file
# video_path = 'C:\\Users\\kooo\\Downloads\\New_Sample\\ja-ma\\9_12\\video\\2021-08-01_09-12-00_sun_sunny_out_ja-ma_C0041.mp4'

# # Initialize video capture with the video file
# cap = cv2.VideoCapture(video_path)

# def compute_velocity_and_acceleration(positions):
#     # Calculate velocities
#     velocities = np.diff(positions, axis=0)
#     # Calculate accelerations
#     accelerations = np.diff(velocities, axis=0)
#     return velocities, accelerations

# # Dictionary to store positions and outputs for each person
# #positions = {}
# positions = {}
# final_output = {}

# try:
#     while True:
        
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Convert the captured frame to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Perform object detection
#         results = model([rgb_frame], size=640)  # Process the image for detection
        
#         # Convert results to DataFrame for easier manipulation
#         df = results.pandas().xyxy[0]  # Results as DataFrame

#         # Filter detections for class 'person' (class ID for 'person' is typically 0 for COCO dataset)
#         person_df = df[df['class'] == 0]  # Assuming 'person' class ID is 0

#         for index, row in person_df.iterrows():
#             person_id = f"person_{index}"  # Create a unique ID for each person
#             x_center = (row['xmin'] + row['xmax']) / 2
#             y_center = (row['ymin'] + row['ymax']) / 2

#             if person_id not in positions:
#                 positions[person_id] = []

#             positions[person_id].append([x_center, y_center])
        
#             if len(positions[person_id]) > 2:
#                 velocities, accelerations = compute_velocity_and_acceleration(np.array(positions[person_id]))
#                 if len(velocities) > 0 and len(accelerations) > 0:
#                     latest_velocity = velocities[-1]
#                     latest_acceleration = accelerations[-1]
#                     final_output[f'PEDESTRIAN/{person_id}'] = np.array([[x_center, y_center, *latest_velocity, *latest_acceleration]])
#                     print(final_output)
#         # Render results back on the frame (modifies the images in-place)
#         results.render()

#         # Convert RGB image to BGR for OpenCV display
#         output_frame = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)

#         # Display the frame
#         cv2.imshow('YOLOv5 Object Detection', output_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# except KeyboardInterrupt:
#     print("Stream stopped")
# finally:
#     cap.release()
#     cv2.destroyAllWindows()





# import cv2
# import torch
# import numpy as np

# # Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Path to your video file
# video_path = 'C:\\Users\\kooo\\Downloads\\New_Sample\\ja-ma\\9_12\\video\\2021-08-01_09-12-00_sun_sunny_out_ja-ma_C0041.mp4'

# # Initialize video capture with the video file
# cap = cv2.VideoCapture(video_path)

# def compute_velocity_and_acceleration(positions):
#     # Calculate velocities
#     velocities = np.diff(positions, axis=0)
#     # Calculate accelerations
#     accelerations = np.diff(velocities, axis=0)
#     return velocities, accelerations

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert the captured frame to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Perform object detection
#         results = model([rgb_frame], size=640)  # Process the image for detection
        
#         # Convert results to DataFrame for easier manipulation
#         df = results.pandas().xyxy[0]  # Results as DataFrame

#         # Initialize dictionaries for storing positions and output data for each person
#         positions = {}
#         final_output = {}

#         # Filter detections for class 'person' (class ID for 'person' is typically 0 for COCO dataset)
#         person_df = df[df['class'] == 0]  # Assuming 'person' class ID is 0

#         for index, row in person_df.iterrows():
#             person_id = f"person_{index}"  # Create a unique ID for each person
#             x_center = (row['xmin'] + row['xmax']) / 2
#             y_center = (row['ymin'] + row['ymax']) / 2

#             if person_id not in positions:
#                 positions[person_id] = []

#             positions[person_id].append([x_center, y_center])
        
#             if len(positions[person_id]) > 2:
#                 velocities, accelerations = compute_velocity_and_acceleration(np.array(positions[person_id]))
#                 if len(velocities) > 0 and len(accelerations) > 0:
#                     latest_velocity = velocities[-1]
#                     latest_acceleration = accelerations[-1]
#                     final_output[f'PEDESTRIAN/{person_id}'] = np.array([[x_center, y_center, *latest_velocity, *latest_acceleration]])
#             print("ddd")

#         # Render results back on the frame (modifies the images in-place)
#         results.render()

#         # Convert RGB image to BGR for OpenCV display
#         output_frame = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)

#         # Display the frame
#         cv2.imshow('YOLOv5 Object Detection', output_frame)

#         # Print final output for the current frame
#         for key, value in final_output.items():
#             print(f"{key}: {value}")

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# except KeyboardInterrupt:
#     print("Stream stopped")
# finally:
#     cap.release()
#     cv2.destroyAllWindows()



import cv2
import torch
import numpy as np
import time

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

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Initialize positions dictionary for each new frame
        positions = {}

        # Convert the captured frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model([rgb_frame], size=640)  # Process the image for detection
        
        # Convert results to DataFrame for easier manipulation
        df = results.pandas().xyxy[0]  # Results as DataFrame

        # Filter detections for class 'person' (class ID for 'person' is typically 0 for COCO dataset)
        person_df = df[df['class'] == 0]  # Assuming 'person' class ID is 0

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
                    final_output[f'PEDESTRIAN/{person_id}'] = np.array([[x_center, y_center, *latest_velocity, *latest_acceleration]])
        
        # Optional: Print final output for debugging
        # for key, value in final_output.items():
        #     print(f"{key}: {value}")

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
