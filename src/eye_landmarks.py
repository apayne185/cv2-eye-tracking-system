import cv2
import mediapipe as mp
import numpy as np 
from gaze_analysis import generate_heatmap


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam successfully opened!")

# eye landmarks from mediapipe facemesh model 
LEFT_EYE_LANDMARKS = [33, 133, 160, 158, 159, 144, 145, 153]  
RIGHT_EYE_LANDMARKS = [362, 263, 387, 385, 386, 374, 380, 373]

def draw_eye_lm(frame, face_landmarks, eye_landmarks, color): 
    h, w, _ = frame.shape
    points = []

    for lm_id in eye_landmarks:
        landmark = face_landmarks.landmark[lm_id]
        x,y = int(landmark.x * w), int(landmark.y *h) 
        points.append((x,y))
        cv2.circle(frame, (x,y),2, color, -1)

    if len(points) > 1: 
        cv2.polylines(frame, [np.array(points, np.int32)], isClosed=True, color=color, thickness=1)




def get_eye_center(face_landmarks, eye_landmarks, frame_shape):            
    h, w, _ = frame_shape  
    x_sum, y_sum = 0, 0  

    for lm_id in eye_landmarks:  
        landmark = face_landmarks.landmark[lm_id]    
        x_sum+= landmark.x * w  
        y_sum+= landmark.y * h 


    return int(x_sum / len(eye_landmarks)), int(y_sum / len(eye_landmarks))    #avergaed position


 

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #convert frames to RGB for mediapipe
    #detects facial landmarks 
    results = face_mesh.process(rgb_frame)  
    gaze_points = []  

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks: 
            draw_eye_lm(frame, face_landmarks, LEFT_EYE_LANDMARKS, (0, 255, 0))          #left eye green
            draw_eye_lm(frame, face_landmarks, RIGHT_EYE_LANDMARKS, (255, 0, 0))          #right eye blue 

            left_eye_center = get_eye_center(face_landmarks, LEFT_EYE_LANDMARKS, frame.shape)
            right_eye_center = get_eye_center(face_landmarks, RIGHT_EYE_LANDMARKS, frame.shape)

            gaze_x = (left_eye_center[0] + right_eye_center[0]) // 2            #avg of both eyes
            gaze_y = (left_eye_center[1] + right_eye_center[1]) // 2
            gaze_points.append((gaze_x, gaze_y))

            cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1) 

     
    if len(gaze_points) > 1:
        heatmap = generate_heatmap(frame, gaze_points)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        cv2.imshow("Eye Gaze Heatmap", overlay)
    else:
        cv2.imshow("Eye Gaze Heatmap", frame)


    # cv2.imshow("Eye landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):      #press q to quit
        break 


cap.release()
cv2.destroyAllWindows()  


