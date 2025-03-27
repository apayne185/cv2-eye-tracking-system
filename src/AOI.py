import cv2
from collections import defaultdict
import time

# Define AOIs (Areas of Interest) as rectangles (x1, y1, x2, y2)
AOIs = {
    "Left Sidebar": (50, 100, 300, 400),  # Example coordinates
    "Product Area": (320, 100, 600, 400),
    "Right Sidebar": (610, 100, 900, 400),
}

# Initialize the dictionary to track time spent in each AOI
aoi_time_spent = defaultdict(float)
last_gaze_time = None

# Function to check if gaze is within a given AOI
def check_aoi(gaze_point, aoi):
    x, y = gaze_point
    x1, y1, x2, y2 = aoi
    return x1 <= x <= x2 and y1 <= y <= y2

# Function to draw AOIs on the frame
def draw_aoi(frame, aoi, color, label):
    x1, y1, x2, y2 = aoi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Function to track time spent in each AOI
def track_aoi(frame, gaze_point):
    global last_gaze_time, aoi_time_spent

    current_time = time.time()

    # Only update if gaze is inside an AOI
    for aoi_name, aoi in AOIs.items():
        if check_aoi(gaze_point, aoi):
            if last_gaze_time is not None:
                aoi_time_spent[aoi_name] += current_time - last_gaze_time
            draw_aoi(frame, aoi, (0, 255, 0), aoi_name)  # Draw AOI on frame

    # Update last_gaze_time with the current time
    last_gaze_time = current_time

# Function to print AOI time stats
def print_aoi_time():
    print("Time spent in each AOI:")
    for aoi_name, time_spent in aoi_time_spent.items():
        print(f"{aoi_name}: {time_spent:.2f} seconds")