import cv2
import mediapipe as mp
import numpy as np 
from gaze_analysis import generate_heatmap


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
 

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
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



#MAIN CODE
def main():
    heatmap_accum = None


    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Exiting...")
        return
    frame_height, frame_width = frame.shape[:2]
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'XVID' or other codecs
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (frame_width, frame_height))
    

    while True: 
        ret, frame = cap.read()
        if not ret:
            break
            # Initialize the heatmap accumulator once we know frame size
        if heatmap_accum is None:
            heatmap_accum = np.zeros(frame.shape[:2], dtype=np.float32)

        # ADD THESE LINES TO FORCE A VALUE IN THE CENTER EVERY FRAME
        center_y = frame.shape[0] // 2
        center_x = frame.shape[1] // 2
        heatmap_accum[center_y, center_x] += 100  # ARTIFICIAL TEST INJECTION

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #convert frames to RGB for mediapipe
        #detects facial landmarks 
        results = face_mesh.process(rgb_frame)  
        #gaze_points = []  


        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:  
                draw_eye_lm(frame, face_landmarks, LEFT_EYE_LANDMARKS, (0, 255, 0))          #left eye green
                draw_eye_lm(frame, face_landmarks, RIGHT_EYE_LANDMARKS, (255, 0, 0))          #right eye blue 

                left_eye_center = get_eye_center(face_landmarks, LEFT_EYE_LANDMARKS, frame.shape)
                right_eye_center = get_eye_center(face_landmarks, RIGHT_EYE_LANDMARKS, frame.shape)

                gaze_x = (left_eye_center[0] + right_eye_center[0]) // 2            #avg of both eyes
                gaze_y = (left_eye_center[1] + right_eye_center[1]) // 2 
                # Accumulate gaze point in heatmap array
                if 0 <= gaze_x < frame.shape[1] and 0 <= gaze_y < frame.shape[0]:
                    heatmap_accum[gaze_y, gaze_x] += 50
                    print("Frame shape:", frame.shape)
                    print("Accumulator shape:", heatmap_accum.shape)
                    print("Max in accumulator:", np.max(heatmap_accum))
    
                # Draw the gaze point on the frame.
                cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 0, 255), -1)

            



        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):      #press q to quit
         exit() 

        elif key == ord('c'):    # PRESS C TO CLEAR THE ACCUMULATOR
            heatmap_accum = np.zeros(frame.shape[:2], dtype=np.float32)  # CHANGE: RESET ACCUMULATOR

    cap.release()
    out.release() 
    cv2.destroyAllWindows()  
    # --------- Produce a Final Standalone Heatmap Image ---------
    # Create a white background image.
    white_background = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    # Regenerate the final heatmap color image from the accumulator.
    final_heatmap_color = generate_heatmap(heatmap_accum)
    # Blend the heatmap with the white background.
    final_overlay = cv2.addWeighted(white_background, 0.5, final_heatmap_color, 0.5, 0)
    # Save and display the final heatmap image.
    cv2.imwrite("final_gaze_heatmap.png", final_overlay)
    cv2.imshow("Final Gaze Heatmap", final_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


