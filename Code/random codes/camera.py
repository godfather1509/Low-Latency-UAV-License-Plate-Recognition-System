import cv2

# === Settings ===
width, height = 640, 480
fps = 15
output_filename = "recorded_output.avi"

# === Open Camera ===
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use CAP_DSHOW for USB cams on Windows

# Set resolution and frame rate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

# === Output video writer ===
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

print("Press 'q' or close the window to exit.")

# === Main Video Capture Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    # Write the frame to the output video file
    out.write(frame)

    # Show the frame in the window
    cv2.imshow("USB2.0 PC CAMERA Feed", frame)
    # "ffmpeg -list_devices true -f dshow -i dummy" use this command to list names of all input devices 
    # add the name of device you want to see output of here
    # Exit the loop if 'q' is pressed or window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("USB2.0 PC CAMERA Feed", cv2.WND_PROP_AUTOSIZE) < 0:
        break

# === Cleanup ===
cap.release()  # Release the camera
out.release()  # Release the video writer
cv2.destroyAllWindows()  # Close any OpenCV windows
