import cv2
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

# eye landmarks from mediapipe facemesh model 
LEFT_EYE_LANDMARKS = [33, 133, 160, 158, 159, 144, 145, 153]  
RIGHT_EYE_LANDMARKS = [362, 263, 387, 385, 386, 374, 380, 373]

while True: 
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #convert frames to RGB for mediapipe
    #detects facial landmarks 
    results = face_mesh.process(rgb_frame)    

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks: 
            h,w, _ = frame.shape
            for landmark_id in LEFT_EYE_LANDMARKS+RIGHT_EYE_LANDMARKS:
                landmark = face_landmarks.landmark[landmark_id] 
                x, y = int(landmark.x*w), int(landmark.y * h)   
                cv2.circle(frame, (x,y), 1, (0, 255, 0), -1)      # draw landmark points


    cv2.imshow("Eye landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):      #press q to quit
        break 


cap.release()
cv2.destroyAllWindows()  


