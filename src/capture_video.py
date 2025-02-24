import cv2

# Capture the video from the camera

cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):    #press q to quit
        break 


cap.release()
cv2.destroyAllWindows()  




