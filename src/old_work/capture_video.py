# import cv2

# # Capture the video from the camera
# # we only use this if we dont want realtime face tracking , it records a video

# def capture_video(output_path='video.mp4', frame_width=1280, frame_height=720, fps=30):
#     cap = cv2.VideoCapture(0)
#     cap.set(3, frame_width)
#     cap.set(4, frame_height) 
 
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))   #output data

#     #while camera/video is running
#     while cap.isOpened(): 
#         ret, frame = cap.read()
#         if not ret: 
#             break 


#         out.write(frame)  
#         cv2.imshow("Video Feed", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):    #press q to quit
#             break 


#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()  
#     print(f"Video saved as {output_path}")


# if __name__ == "__main__":
#     capture_video()



import cv2

cap = cv2.VideoCapture(0)  # Open default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    cv2.imshow("Webcam Feed", frame)  # Show webcam feed

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

