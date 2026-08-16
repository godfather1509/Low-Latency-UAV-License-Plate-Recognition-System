import os
from ultralytics import YOLO
import cv2

video_path = os.path.join('.', 'Data/testVideo.mp4')
video_path_out = f'{video_path}_out.mp4'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret or frame is None:
    print("Error: Could not read the video.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'),
                      int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(),
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (0, 255, 0), 3)

    # ✔ Write to output file
    out.write(frame)

    # ✔ Display video
    cv2.imshow("YOLO Detection", frame)

    # ✔ Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read next frame
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
