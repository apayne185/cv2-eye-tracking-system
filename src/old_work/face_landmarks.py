import cv2
import mediapipe as mp


#unsure if we will need this or not since we are just focusing on eye tracking

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)          #suggested for a more natural webcam view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #convert frames to RGB for mediapipe
    #detects facial landmarks 
    results = face_mesh.process(rgb_frame)    

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks: 
            #draws the landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)  

            # for landmark in face_landmarks.landmarks: 
                # h,w, _ = frame.shape
                # x, y = int(landmark.x*w), int(landmark.y * h)   
                # cv2.circle(frame, (x,y), 1, (0, 255, 0), -1)      # draw landmark points


    cv2.imshow("Face landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):      #press q to quit
        break 


cap.release()
cv2.destroyAllWindows()  


